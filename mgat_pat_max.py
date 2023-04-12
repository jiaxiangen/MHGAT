import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
class MultiAttention(nn.Module):
    def __init__(self, in_size, hidden_size=64,dropout =0.2):
        super(MultiAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(in_size, hidden_size),
            nn.Dropout(dropout),
            nn.Tanh(),

            nn.Linear(hidden_size, 1),
            nn.Dropout(dropout),
            nn.Tanh()
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                # torch.nn.init.xavier_uniform(m.bias.data)

        # self.project.apply(init_weights)

    def forward(self, z):
        w = self.project(z)
        # print(w[0])
        beta = torch.softmax(w, dim=1)
        if beta.shape[0]>=6:
            print("beta{}".format(beta.shape))
            print(beta[5,:,:])

        return (beta * z).sum(1)



class EdgeGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim,dropout = 0.2,residual = 0.12):
        super(EdgeGATLayer, self).__init__()
        # equation (1)
        self.residual = residual
        self.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(in_dim, out_dim, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
        # nn.Dropout(0.6),
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)

        # self.fc.apply(init_weights)
        # equation (2)
        self.attn_fc = nn.Sequential(
            nn.Linear(2 * out_dim, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2)
            # nn.Tanh()
        )
        self.drop = nn.Dropout(0.1)
        # self.attn_fc.apply(init_weights)

    def forward(self, src_h,h):
        # equation (1)
        src_z = self.fc(src_h)
        h_z = self.fc(h)
        h_t =h_z
        h_z = h_z.unsqueeze(dim=1)
        h_z = torch.repeat_interleave(h_z, src_h.shape[1], dim=1)
# 186
        z2 = torch.cat([src_z, h_z], dim=2)
        z_f = self.attn_fc(z2)

        # equation (3) & (4)

        alpha = F.softmax(z_f, dim=1)
        alpha=self.drop(alpha)
        msg = torch.sum(alpha * src_z, dim=1)
        msg = msg + self.residual*h_t

        return msg



class MMGATLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, bias=None,
                 activation=None,dropout=0.4,residual = 0.12):
        super(MMGATLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation
        self.residual = residual
# 18655836837


        # weight bases in equation (3)
        # weight bases in equation (3)

        self.batchNorm =nn.Sequential(
            nn.BatchNorm1d(num_features=(self.out_feat)),
        )
        self.batchNorm1 = nn.Sequential(
            nn.BatchNorm1d(num_features=(self.in_feat)),
        )
        self.project= nn.Sequential(
            nn.Linear(self.in_feat, self.out_feat),
            nn.Dropout(dropout),
            nn.Tanh(),


        )

        self.attention = nn.ModuleList()
        for i in range(2):
            self.attention.append(EdgeGATLayer(self.in_feat, self.out_feat, dropout=dropout,residual = self.residual))

        # init trainable parameters
        # nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        self.multiAttention = MultiAttention(self.out_feat,hidden_size=64,dropout=dropout)
        # self.reset_parameters()

    def forward(self, g):

        # g.ndata['h'] = self.batchNorm(g.ndata['h'])
        def message_func(edges):
            # w = weight[edges.data['rel_type']]
            # msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            # msg = msg * edges.data['norm']
            # 'msg': msg,
            return {'rel_type':edges.data['rel_type'],'src_h':edges.src['h'], 'src_h_img':edges.src['h_img']}

        def reduce_func(nodes):
            nmb = nodes.mailbox
            if len(nodes.nodes())>=6:
                print(nodes.nodes()[5])

            src_h = nmb['src_h']
            src_h_img = nmb['src_h_img']

            rel_type = nmb['rel_type']
            nodes_h = nodes.data['h']
            nodes_h_img = nodes.data['h_img']
            zero = torch.zeros_like(rel_type)

            one = torch.ones_like(rel_type)
            attention_list = []
            attention_list_img = []
            for data_index in range(self.num_rels):


                mask = torch.where(rel_type != data_index, zero, rel_type)
                mask = torch.where(rel_type == data_index, one, mask)
                # print("mask.shape{}".format(mask.shape))
                mask = mask.unsqueeze(dim=2)

                mask = mask.expand_as(src_h)
                result = torch.mul(src_h, mask)

                mask_img = mask.expand_as(src_h_img)
                result_img = torch.mul(src_h_img,mask_img)


                attention_fin=result.max(dim=1).values
                attention_fin_img = result_img.max(dim=1).values
                # print("attention_fin.shape{}
                attention_list.append(attention_fin.unsqueeze(dim=1))
                attention_list_img.append(attention_fin_img.unsqueeze(dim=1))

            #Text_attention
            attention_fin = torch.cat(attention_list,dim = 1)
            attention_fin = self.attention[0](attention_fin,nodes_h)
            # attention_fin = torch.sum(attention_fin,dim=1)
            # print(attention_fin.shape)
            msg = attention_fin.unsqueeze(dim=1)

            #Img_attention
            attention_fin_img = torch.cat(attention_list_img,dim=1)
            attention_fin_img = self.attention[1](attention_fin_img, nodes_h_img)
            # attention_fin_img = torch.sum(attention_fin_img,dim=1)

            msg_img = attention_fin_img.unsqueeze(dim=1)
            msg_all = torch.cat([msg,msg_img],dim=1)
            multi_msg =self.multiAttention(msg_all)
            return {'h':msg.squeeze(dim=1),'h_img':msg_img.squeeze(dim=1),'multi_msg':multi_msg}


        def apply_func(nodes):
            h = nodes.data['h']

            h = self.batchNorm(h)
            h_img = nodes.data['h_img']
            h_img = self.batchNorm(h_img)
            multi_msg = nodes.data['multi_msg']
            multi_msg = self.batchNorm(multi_msg)
            # multi_msg = F.normalize(multi_msg)
            # h = self.batchNorm(h)
            if self.activation:
                h = self.activation(h)
                h_img = self.activation(h_img)


            return {'h': h,'h_img':h_img,'multi_msg':multi_msg}

        g.update_all(message_func, reduce_func, apply_func)
        # print("jia")
        # print(g.ndata['h'])
        return g.ndata['multi_msg']