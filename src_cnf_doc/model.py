import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

import layers


class MVNLogProb(nn.Module):

    def __init__(self):
        super(MVNLogProb, self).__init__()

    def forward(self, x, mu):
        batch_size = x.shape[0]
        emb_size = x.shape[1]
        diff = (x - mu).view(batch_size, 1, emb_size)
        return -0.5 * diff.bmm(diff.transpose(1, 2)).squeeze() - emb_size/2 * math.log(2 * math.pi)


class MVNLogProbBatched(nn.Module):

    def __init__(self):
        super(MVNLogProbBatched, self).__init__()

    def forward(self, x, mu, ntoken):
        batch_size = x.shape[0]
        emb_size = x.shape[1]
        mu = mu.repeat(1, ntoken).view(-1, emb_size)

        diff = (x - mu).view(batch_size, 1, emb_size)
        return -0.5 * diff.bmm(diff.transpose(1, 2)).squeeze() - emb_size/2 * math.log(2 * math.pi)


class ODEnet(nn.Module):

    def __init__(self, dim):
        super(ODEnet, self).__init__()
        self.linear1 = layers.diffeq_layers.ConcatLinear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.soft_relu = nn.Softplus()

        # self.init_weights()

    def forward(self, t, x):

        out = self.linear1(t, x)
        out = self.soft_relu(out)
        out = self.linear2(out)

        return out


class CNFBlock(nn.Module):

    def __init__(self, ninp, ntoken):
        super(CNFBlock, self).__init__()

        def build_cnf():
            # diffeq = layers.ODEnet(
            #     hidden_dims=(ninp,),
            #     input_shape=(ninp,),
            #     strides=None,
            #     conv=False,
            #     layer_type='concat',
            #     nonlinearity='softplus'
            # )

            diffeq = ODEnet(ninp)

            odefunc = layers.ODEfunc(
                diffeq=diffeq,
                # divergence_fn=args.divergence_fn,
                # residual=args.residual,
                # rademacher=args.rademacher,
            )

            cnf = layers.CNF(
                odefunc=odefunc,
                # T=args.time_length,
                # train_T=args.train_T,
                # regularization_fns=None,
                # solver='rk4',
            )
            return cnf

        self.ninp = ninp
        self.ntoken = ntoken
        self.mvn_log_prob = MVNLogProb()
        self.mvn_log_prob_batched = MVNLogProbBatched()
        self.cnf = build_cnf()

    def forward(self, h, emb_matrix, log_pz0=None):

        seq_length, batch_size, emb_size = h.shape
        h = h.view(seq_length * batch_size, emb_size)

        # FULL SOFTMAX SAME TRANSFORMATION BATCHED
        # When the transformation does not depend on the hidden state only one CNF
        # can be solved instead of seq_length * batch_size CNFs

        z0 = emb_matrix
        zeros = torch.zeros(self.ntoken, 1).to(z0)
        z1, delta_log_pz = self.cnf(z0, zeros)

        # This can be batched, but as memory is bigger issue this is more optimal
        if log_pz0 is None:
            for h_i in h:

                if log_pz0 is None:
                    log_pz0 = self.mvn_log_prob(z0, h_i)
                else:
                    log_pz0 = torch.cat((log_pz0, self.mvn_log_prob(z0, h_i)))

        log_pz1 = log_pz0.view(-1, 1) - delta_log_pz.repeat(seq_length * batch_size, 1)
        log_pz1 = log_pz1.view(-1, self.ntoken)

        return log_pz1


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nhidlast, nlayers,
                 decoder_log_pz0,
                 dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0,
                 tie_weights=False, ldropout=0.6,
                 n_experts=10, num4embed=0, num4first=0, num4second=0,
                 use_dropout=True):
        super(RNNModel, self).__init__()

        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)

        self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else nhidlast, 1, dropout=0) for l
                     in range(nlayers)]
        if wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.all_experts = n_experts + num4embed + num4first + num4second
        self.prior = nn.Linear(nhidlast, self.all_experts, bias=False)
        self.latent = nn.Linear(nhidlast, n_experts * ninp)
        if num4embed > 0:
            self.weight4embed = nn.Linear(ninp, num4embed * ninp)
        if num4first > 0:
            self.weight4first = nn.Linear(nhid, num4first * ninp)
        if num4second > 0:
            self.weight4second = nn.Linear(nhid, num4second * ninp)
        self.decoder = nn.Linear(ninp, ntoken)
        self.cnf = CNFBlock(ninp, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            # if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.num4embed = num4embed
        self.num4first = num4first
        self.num4second = num4second
        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.dropoutl = ldropout
        self.n_experts = n_experts
        self.decoder_log_pz0 = decoder_log_pz0
        self.ntoken = ntoken

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('param size: {}'.format(size))

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.latent.bias.data.fill_(0)
        if self.num4embed > 0:
            self.weight4embed.bias.data.fill_(0)
        if self.num4first > 0:
            self.weight4first.bias.data.fill_(0)
        if self.num4second > 0:
            self.weight4second.bias.data.fill_(0)

    def forward(self, input, hidden, return_h=False, return_prob=False):
        batch_size = input.size(1)

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        # emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)
        list4mos = []
        if self.num4embed > 0:
            embed4mos = nn.functional.tanh(self.weight4embed(emb))
            embed4mos = embed4mos.view(emb.size(0), emb.size(1), self.num4embed, self.ninp).transpose(1, 2).transpose(1,
                                                                                          0).contiguous()
            embed4mos = embed4mos.view(-1, emb.size(1), self.ninp)
            list4mos.extend(list(torch.chunk(embed4mos, self.num4embed, 0)))

        raw_output = emb
        new_hidden = []
        # raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                # self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
                if l == 0 and self.num4first > 0:
                    first4mos = nn.functional.tanh(self.weight4first(raw_output))
                    first4mos = first4mos.view(raw_output.size(0), raw_output.size(1), self.num4first,
                                               self.ninp).transpose(1, 2).transpose(1, 0).contiguous()
                    first4mos = first4mos.view(-1, raw_output.size(1), self.ninp)
                    list4mos.extend(list(torch.chunk(first4mos, self.num4first, 0)))
                if l == 1 and self.num4second > 0:
                    second4mos = nn.functional.tanh(self.weight4second(raw_output))
                    second4mos = second4mos.view(raw_output.size(0), raw_output.size(1), self.num4second,
                                                 self.ninp).transpose(1, 2).transpose(1, 0).contiguous()
                    second4mos = second4mos.view(-1, raw_output.size(1), self.ninp)
                    list4mos.extend(list(torch.chunk(second4mos, self.num4second, 0)))
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        latent = nn.functional.tanh(self.latent(output))
        # apply same mask to all context vec
        transd = latent.view(
            raw_output.size(0), raw_output.size(1), self.n_experts, -1
        ).transpose(1, 2).transpose(1, 0).contiguous().view(-1, raw_output.size(1), self.ninp)

        list4mos.extend(list(torch.chunk(transd, self.n_experts, 0)))
        concated = torch.cat(list4mos, 1)
        dropped = self.lockdrop(concated.view(-1, raw_output.size(1), self.ninp), self.dropoutl)
        contextvec = dropped.view(
            raw_output.size(0), self.all_experts, raw_output.size(1), self.ninp).transpose(1, 2).contiguous()
        logit = self.decoder(contextvec.view(-1, self.ninp))

        prior_logit = self.prior(output).view(-1, self.all_experts)
        prior = nn.functional.softmax(prior_logit)

        prob = nn.functional.softmax(logit.view(-1, self.ntoken)).view(-1, self.all_experts, self.ntoken)
        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)

        ############################################################
        # Continuous Normalizing Flows
        ############################################################

        if self.decoder_log_pz0:
            log_pz0 = torch.log(prob.add_(1e-8))
        else:
            log_pz0 = None

        log_pz1 = self.cnf(output, self.encoder.weight, log_pz0)
        prob = nn.functional.softmax(log_pz1, -1)

        ############################################################

        if return_prob:
            model_output = prob
        else:
            log_prob = torch.log(torch.add(prob, 1e-8))
            model_output = log_prob

        model_output = model_output.view(-1, batch_size, self.ntoken)
        prior = prior.view(-1, batch_size, self.all_experts)

        if return_h:
            return model_output, hidden, raw_outputs, outputs, prior
        return model_output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()),
                 Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()))
                for l in range(self.nlayers)]


if __name__ == '__main__':
    model = RNNModel('LSTM', 10, 12, 12, 12, 2)
    input = Variable(torch.LongTensor(13, 9).random_(0, 10))
    print('input', input)
    hidden = model.init_hidden(9)
    model(input, hidden)

    # input = Variable(torch.LongTensor(13, 9).random_(0, 10))
    # hidden = model.init_hidden(9)
    # print(model.sample(input, hidden, 5, 6, 1, 2, sample_latent=True).size())
