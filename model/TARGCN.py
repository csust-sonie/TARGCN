import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from model.GRU import GRU
from model.temporal_attention_layer import transformer_layer
from torch.autograd import Variable
import math
device=torch.device('cuda')


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, adj,num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one GRU layer in the Encoder.'
        self.adj=adj
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(GRU(node_num, dim_in, dim_out, self.adj,cheb_k, embed_dim))
        # self.tcn=TemporalConvNet(dim_in,[1,1,1],3,0.2)
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(GRU(node_num, dim_out, dim_out,self.adj ,cheb_k, embed_dim))
        self.trans_layer_T = transformer_layer(dim_out, dim_out, 2, 2)

    def forward(self, x, init_state, node_embeddings):

        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]

        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        current_inputs=self.trans_layer_T(current_inputs)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class TARGCN(nn.Module):
    def __init__(self, args,adj=None):
        super(TARGCN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.adj=adj
        # self.default_graph = args.default_graph

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim,self.adj, args.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(6, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        # self.FC = nn.Linear(6,6)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -6:, :, :]
        output = self.end_conv((output))                         #B, T*C, N, 1


        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node) # b t c n
        output = output.permute(0, 1, 3, 2)                             #B, T(12), N, C

        return output

if __name__=='__main__':
    import argparse
    import configparser
    config = configparser.ConfigParser()
    config_file = 'PEMSD8.conf'

    config.read(config_file)

    args = argparse.ArgumentParser(description='arguments')

    args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    args.add_argument('--horizon', default=config['data']['horizon'], type=int)
    args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
    # args.add_argument('--embed_dim', default=2, type=int)
    args = args.parse_args()

    num_node = args.num_nodes
    input_dim = args.input_dim
    hidden_dim = args.rnn_units
    output_dim = args.output_dim
    horizon = args.horizon
    num_layers = args.num_layers
    adj = torch.ones((num_node,num_node))
    # print(adj.shape)
    node_embeddings = nn.Parameter(torch.randn(num_node, 2), requires_grad=True)
    agcrn=AGCRN(args,adj)
    # source: B, T_1, N, D
    # target: B, T_2, N, D
    x=torch.randn(32,12,170,1)
    tar=torch.randn(32,12,170,1)
    out=agcrn(x,tar)
    print(out.shape)
