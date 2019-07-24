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
            print('test')
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
                 tie_weights=False, ldropout=0.5, use_dropout=True):
        super(RNNModel, self).__init__()

        self.use_dropout = use_dropout
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)

        self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else nhidlast, 1, dropout=0) for l
                     in range(nlayers)]

        if wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop if self.use_dropout else 0) for rnn in
                         self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)

        if nhidlast != ninp:
            self.latent = nn.Sequential(nn.Linear(nhidlast, ninp), nn.Tanh())

        self.decoder = nn.Linear(ninp, ntoken)
        self.cnf = CNFBlock(ninp, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhidlast != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.nlayers = nlayers
        self.decoder_log_pz0 = decoder_log_pz0
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.ldropout = ldropout
        self.dropoutl = ldropout
        self.ntoken = ntoken

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('param size: {}'.format(size))

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False, return_prob=False, sampled_targets=None):

        batch_size = input.size(1)

        emb = embedded_dropout(self.encoder, input,
                               dropout=self.dropoute if (self.training and self.use_dropout) else 0)
        # emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti if self.use_dropout else 0)

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
                raw_output = self.lockdrop(raw_output, self.dropouth if self.use_dropout else 0)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout if self.use_dropout else 0)
        outputs.append(output)

        if self.nhidlast != self.ninp:
            output = self.latent(output)

        ############################################################
        # Regular LM
        ############################################################
        #
        # logit = self.decoder(output)
        # prob = nn.functional.softmax(logit, -1)
        #
        ############################################################

        ############################################################
        # Continuous Normalizing Flows
        ############################################################

        # if self.decoder_log_pz0:
        #     log_pz0 = self.decoder(output)
        # else:
        #     log_pz0 = None

        log_pz0 = self.decoder(output)
        log_pz1 = self.cnf(output, self.encoder.weight, log_pz0)
        # log_pz1 = log_pz0
        # print('log_pz1 shape', log_pz1.shape)
        prob = nn.functional.softmax(log_pz1, -1)

        ############################################################

        if return_prob:
            model_output = prob
        else:
            log_prob = torch.log(torch.add(prob, 1e-8))
            model_output = log_prob

        # print('model_output shape', model_output.shape)

        # FULL SOFTMAX
        if sampled_targets is None:
            model_output = model_output.view(-1, batch_size, self.ntoken)
        else:
            model_output = model_output.view(-1, batch_size, sampled_targets.size(1))

        if return_h:
            return model_output, hidden, raw_outputs, outputs
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
