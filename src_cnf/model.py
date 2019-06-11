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


class CNFBlock(nn.Module):

    def __init__(self, ninp, ntoken):
        super(CNFBlock, self).__init__()

        def build_cnf():
            diffeq = layers.ODEnet(
                hidden_dims=(ninp,),
                input_shape=(ninp,),
                strides=None,
                conv=False,
                layer_type='concat',
                nonlinearity='softplus'
            )
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
                # solver=args.solver,
            )
            return cnf

        self.ninp = ninp
        self.ntoken = ntoken
        self.mvn_log_prob = MVNLogProb()
        self.mvn_log_prob_batched = MVNLogProbBatched()
        self.cnf = build_cnf()

    def forward(self, h, emb_matrix, sampled_targets):

        num_sampled = sampled_targets.size(1)
        seq_length, batch_size, emb_size = h.shape
        h = h.view(seq_length * batch_size, emb_size)

        # FULL BATCHED
        # z0 = emb_matrix.repeat(seq_length * batch_size, 1)
        # zeros = torch.zeros(seq_length * batch_size * self.ntoken, 1).to(emb_matrix)
        # print('shape of zeros ', zeros.shape)
        # print('size of zeros ', zeros.element_size() * zeros.nelement())
        # _, delta_log_pz = self.cnf(z0, zeros)
        # delta_log_pz = delta_log_pz.view(-1, self.ntoken)
        #
        # log_pz0 = self.mvn_log_prob_batched(z0, h).view(-1, self.ntoken)


        # SAMPLED
        l_z0 = [emb_matrix[targets] for targets in sampled_targets]
        z0 = torch.stack(l_z0).view(-1, emb_size)
        zeros = torch.zeros(seq_length * batch_size * num_sampled, 1).to(z0)

        _, delta_log_pz = self.cnf(z0, zeros)
        delta_log_pz = delta_log_pz.view(-1, num_sampled)

        log_pz0 = self.mvn_log_prob_batched(z0, h, num_sampled).view(-1, num_sampled)

        # print('shape of z0 ', z0.shape)
        # print('size of z0 ', z0.element_size() * z0.nelement())
        # print('zeros shape', zeros.shape)
        # print('delta_log_pz shape', delta_log_pz.shape)
        # print('log_pz0 shape', log_pz0.shape)

        # l_delta_log_pz = []
        # l_log_pz0 = []
        # for i in range(seq_length * batch_size):
        #     print('{} | {}'.format(i, seq_length * batch_size))
        #     sys.stdout.flush()
        #
        #     z0 = emb_matrix
        #     zeros = torch.zeros(self.ntoken, 1).to(emb_matrix)
        #
        #     _, tmp_delta_log_pz = self.cnf(z0, zeros)
        #     l_delta_log_pz.append(tmp_delta_log_pz)
        #
        #     tmp_log_pz0 = self.mvn_log_prob(emb_matrix, h[i])
        #     l_log_pz0.append(tmp_log_pz0)
        #
        # delta_log_pz = torch.stack(l_delta_log_pz).view(-1, self.ntoken)
        # log_pz0 = torch.stack(l_log_pz0).view(-1, self.ntoken)

        log_pz1 = log_pz0 - delta_log_pz

        return log_pz1


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nhidlast, nlayers,
                 dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0,
                 tie_weights=False, ldropout=0.5):
        super(RNNModel, self).__init__()
        self.use_dropout = True
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

        # self.decoder = nn.Linear(ninp, ntoken)
        self.cnf = CNFBlock(ninp, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        # if tie_weights:
            # if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            # self.decoder.weight = self.encoder.weight

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

        log_pz1 = self.cnf(output, self.encoder.weight, sampled_targets)
        prob = nn.functional.softmax(log_pz1, -1)

        ############################################################

        if return_prob:
            model_output = prob
        else:
            log_prob = torch.log(torch.add(prob, 1e-8))
            model_output = log_prob

        # FULL SOFTMAX
        # model_output = model_output.view(-1, batch_size, self.ntoken)
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
