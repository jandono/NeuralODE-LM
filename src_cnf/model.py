import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

# if args.adjoint:
#     from torchdiffeq import odeint_adjoint as odeint
# else:
#     from torchdiffeq import odeint

from torchdiffeq import odeint_adjoint as odeint
import layers
from layers.odefunc import divergence_bf, divergence_approx
from layers.odefunc import sample_gaussian_like, sample_rademacher_like
from layers.diffeq_layers import ConcatLinear
from torch.distributions import MultivariateNormal


class MVNLogProb(nn.Module):

    def __init__(self):
        super(MVNLogProb, self).__init__()

    def forward(self, x, mu):
        batch = x.shape[0]
        k = x.shape[1]
        diff = (x - mu).view(batch, 1, k)
        return torch.squeeze(-0.5 * torch.bmm(diff, torch.transpose(diff, 1, 2)) - k/2 * math.log(2 * math.pi))


class ODEnet(nn.Module):

    def __init__(self, dim):
        super(ODEnet, self).__init__()
        self.cc_linear = ConcatLinear(dim, dim)
        self.relu = nn.ReLU(inplace=True)  # Inplace can cause problems
        # self.nfe = 0

    def forward(self, t, x):
        # For simple cases time can be omitted.
        # However, for CNF they mention that they use a Hypernetwork or Concatenation
        # self.nfe += 1
        out = self.cc_linear(t, x)
        out = self.relu(out)
        return out


class CNFBlock(nn.Module):

    def __init__(self, ninp, ntoken):
        super(CNFBlock, self).__init__()

        def build_cnf(ninp):
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
                # solver=args.solver,
            )
            return cnf

        self.ninp = ninp
        self.ntoken = ntoken
        self.mvn_log_prob = MVNLogProb()
        self.cnf = build_cnf(ninp)

    def forward(self, h, encoder):

        seq_length, batch_size, emb_size = h.shape
        h = h.view(seq_length * batch_size, emb_size)
        # print('h shape', h.shape)

        emb_matrix = encoder.weight
        # print('emb matrix shape', emb_matrix.shape)

        # print('CNF...')

        l_logpz0 = []
        l_delta_logpz = []
        for i in range(seq_length * batch_size):
            zeros = torch.zeros(self.ntoken, 1).to(emb_matrix)
            _, tmp_delta_log_pz = self.cnf(emb_matrix, zeros)
            l_delta_logpz.append(torch.squeeze(tmp_delta_log_pz))

            # mvn = MultivariateNormal(h[i], torch.eye(h[i].size(0))
            # tmp_log_pz0 = mvn.log_prob(emb_matrix)
            # print('tmp_log_pz0 shape', tmp_log_pz0.shape)

            tmp_log_pz0 = self.mvn_log_prob(emb_matrix, h[i])
            l_logpz0.append(tmp_log_pz0)

        # print('CNF Done')

        log_pz0 = torch.stack(l_logpz0).view(-1, self.ntoken)
        # print('log_pz0 shape', log_pz0.shape)

        delta_log_pz = torch.stack(l_delta_logpz)
        # print('delta_log_pz', delta_log_pz.shape)

        log_pz1 = log_pz0 - delta_log_pz
        # print('log_pz1 shape', log_pz1.shape)

        log_pz1 = log_pz1.view(-1, self.ntoken)
        # print('log_pz1 shape', log_pz1.shape)

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
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False, return_prob=False):

        # print('input shape', input.shape)

        seq_length = input.size(0)
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

        # print('output shape', output.shape)

        if self.nhidlast != self.ninp:
            output = self.latent(output)

        ############################################################
        # Continuous Normalizing Flows
        ############################################################

        # print('output shape', output.shape)

        log_pz1 = self.cnf(output, self.encoder)

        # print('log_pz1 shape', log_pz1.shape)
        # assert 1 == 0
        ############################################################

        # logit = self.decoder(output)
        # print('logit shape', logit.shape)
        # assert 1 == 0
        #
        # # transformed = self.ode(logit)
        # transformed = logit

        # converts the log densities to discrete probabilities
        prob = nn.functional.softmax(log_pz1, -1)
        # print('prob shape', prob.shape)

        if return_prob:
            model_output = prob
        else:
            log_prob = torch.log(torch.add(prob, 1e-8))
            model_output = log_prob

        # model_output = log_pz1
        model_output = model_output.view(-1, batch_size, self.ntoken)

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
