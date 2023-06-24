import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing,GCNConv
from torch_geometric.utils import add_self_loops,softmax,degree
from torch_scatter import scatter


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class HighwayMLP(nn.Module):

    def __init__(self,
                 input_size,
                 gate_bias=-2,
                 activation_function=nn.functional.relu,
                 gate_activation=nn.functional.softmax):

        super(HighwayMLP, self).__init__()

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.normal_layer = nn.Linear(input_size, input_size)

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):

        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x),dim=1)

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN,self).__init__()

        self.gcn1 = GCNConv(in_feats, hidden_size)
        self.gcn2 = GCNConv(hidden_size, num_classes)

        self.highway = HighwayMLP(hidden_size)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.highway(x)
        x = F.dropout(x, training=self.training)
        x = self.gcn2(x, edge_index)

        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        device1 = input.device
        W = self.W.to(device1)
        h = torch.matmul(input, W)

        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)

        device2 = a_input
        A = self.a.to(device2)
        e = self.leakyrelu(torch.matmul(a_input, A).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        device3 =e.device
        adj =adj.to(device3)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=1)
        attention = nn.functional.dropout(attention, p=self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        return h_prime
    
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha)

    def forward(self, x, adj):
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = nn.functional.elu(self.out_att(x, adj))
        return nn.functional.log_softmax(x, dim=1)


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d