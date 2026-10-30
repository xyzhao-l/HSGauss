import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MLP_Large(nn.Module):

    def __init__(self, input_dim, hidden, output_dim, dropout=0.2):
        super(MLP_Large, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(F.relu(x))
        x = self.fc2(x)
        x = self.dropout(F.relu(x))
        x = self.fc3(x)
        return x

class MLP(nn.Module):

    def __init__(self, input_dim, hidden, output_dim, dropout=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(F.relu(x))
        x = self.fc3(x)
        return x

class MLP_VEC(nn.Module):

    def __init__(self, input_dim, hidden, output_dim):
        super(MLP_VEC, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.nlnr = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc25 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x, printit=False):
        x = self.fc1(x)
        x = self.nlnr(x)
        x = self.fc2(x)
        x = self.nlnr(x)
        x = self.fc25(x)
        x = self.nlnr(x)
        x = self.fc3(x)
        x = self.sig(x)
        return x

class NormalisedWeights(nn.Module):

    def __init__(self, input_dim, mixturetype=None, modeltype=None):
        super(NormalisedWeights, self).__init__()
        self.type = mixturetype
        self.input_dim = input_dim
        self.valspace = 'complex' if modeltype else 'real'
        if mixturetype == 'random':
            self.weights = nn.Parameter(torch.randn(input_dim))
        elif mixturetype == 'constant':
            self.weights = torch.ones(input_dim).cuda()
        elif mixturetype == 'uniform':
            self.weights = nn.Parameter(torch.empty(input_dim).uniform_(-1, 1))
        elif mixturetype == 'self_attention':
            self.weights = nn.Parameter(torch.empty((input_dim // 2, self.args.matrixsize)))
            self.a = nn.Parameter(torch.empty(input_dim))
        self.fc1 = nn.Softmax(dim=-1)

    def forward(self, x):
        if self.type == 'self_attention':
            weight_mat = self.weights.T
            a = self.a
            weighted_states = torch.matmul(x, weight_mat)
            mat1 = weighted_states.unsqueeze(-2).repeat(1, 1, self.input_dim, 1)
            mat2 = weighted_states.unsqueeze(-3).repeat(1, self.input_dim, 1, 1)
            w_concat = torch.cat([mat1, mat2], dim=-1)
            w_concat = torch.matmul(w_concat, a)
            w_concat = self.lrelu(w_concat)
            weights = self.fc1(w_concat)
            weighted_sum_batch = torch.matmul(weights, weighted_states)
        elif self.type == 'nn':
            weights = self.ann(self.weights)
            weights = self.fc1(weights)
        else:
            weights = self.fc1(self.weights)
            assert x.shape[1] == weights.shape[0]
            if self.valspace == 'complex':
                weights = torch.sqrt(weights)
        return weights

class LINEAR_ONE(nn.Module):

    def __init__(self, input_dim):
        super(LINEAR_ONE, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc2 = nn.Softmax(dim=-1)

    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        x = torch.outer(x, x)
        x = F.linear(x, y)
        return x

class Observation(nn.Module):

    def __init__(self, num_obs=64):
        super(Observation, self).__init__()
        self.weights = nn.Parameter(torch.empty(num_obs, 768).uniform_(-1, 1))

    def forward(self, x):
        output = torch.einsum('ki,bij,kj->bk', self.weights, x, self.weights)
        return output
