import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
copy.deepcopy()
""""
1. gat layer
2. NNPM layer
3. residual layer
4. graph-wsie normalization layer
5. adj-wise normalization layer
6. node-wise normalization layer
7. multi-head layer


网络架构：两层多头注意力的图网络
多跳还是多层的区别在于图网络参数共享不共享


层的多跳传播

层间的多层传播

对每层再做Attention

graph layer中引入gate mechanism

传统attention只能对节点信息编码，如何充分利用上边的信息
"""


class MultiHeadLayer(nn.Module):
    """"
    并行，一层并行出多层
    """
    def __init__(self, w_input_dim, s_input_dim, w_output_dim, s_output_dim, attn_drop, num_heads, gat_layer,
                 merge="cat"):
        super(MultiHeadLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(gat_layer(w_input_dim, s_input_dim, w_output_dim, s_output_dim))
        self.dropout = nn.Dropout(attn_drop)
        self.merge = merge

    def foward(self, g, score):


class GraphAttentionLayer(torch.nn.Module):

    def __init__(self, w_input_dim, s_input_dim, w_output_dim, s_output_dim):  # supernode_output dim和sentence dim相同
        super(GraphAttentionLayer, self).__init__()
        self.s_fc = torch.nn.Linear(s_input_dim, s_output_dim, bias=False)  # 拼接求z时用的
        self.w_fc = torch.nn.Linear(w_input_dim, w_output_dim, bias=False)
        self.super_fc = torch.nn.Linear(s_input_dim, s_output_dim, bias=False)
        self.s_att_fc = torch.nn.Linear(s_input_dim, s_output_dim, bias=False)  # 加权求和时用到的邻居节点隐状态变换
        self.w_att_fc = torch.nn.Linear(w_input_dim, w_output_dim, bias=False)

        self.s_w_attention_fc = torch.nn.Linear(s_output_dim + w_output_dim, 1, bias=False)  # 拼接后进行算分
        self.w_h_fc = torch.nn.Linear(s_output_dim + w_input_dim, w_output_dim, bias=False)  # 邻居信息和自身信息拼接后全联接更新自身信息

        self.s_s_attention_fc = torch.nn.Linear(2 * s_output_dim, 1, bias=False)
        self.w_s_attention_fc = torch.nn.Linear(s_output_dim + w_output_dim, 1, bias=False)
        self.s_h_fc = torch.nn.Linear(s_input_dim + s_output_dim + w_output_dim, s_output_dim, bias=False)

        self.w_super_attention_fc = torch.nn.Linear(s_output_dim + w_output_dim, 1, bias=False)
        self.s_super_attention_fc = torch.nn.Linear(2 * s_output_dim, 1, bias=False)
        self.super_h_fc = torch.nn.Linear(s_input_dim + s_output_dim + w_output_dim,
                                          s_output_dim, bias=False)  # supernode的编码维度和句子相同

    # TODO: 更新图中节点表示: 怎样去聚合邻居节点的信息，怎样针对不同类型的节点进行不同聚合操作以得到聚合后的邻居信息
    # TODO: 得到邻居节点信息后进行怎样的处理来更新自身表示
    # TODO: 得到自身表示后，计算每个节点相对于supernode重要性分数
    # TODO: 返回重要性分数
    # TODO: 有些属性用完后可以pop
    # ？？？G中需要存那些数据？？？
    def forward(self, g, score):
        """

        :param g:
        :param score: [sentence_num, 1] 每个句子相对于supernode的重要性值分数, pad的句子为0分，向量也是零向量
        :return:

        图中节点的属性有：every node：h，z，u，
                      wordnode: n (邻居节点即句子节点聚合后的信息）
                      sentnode：ns (邻居节点即句子节点聚合后的信息） nw(邻居节点即单词节点聚合后的信息)
                      supernode：ns (邻居节点即句子节点聚合后的信息） nw(邻居节点即单词节点聚合后的信息)
        图中边的属性有：every edge：att
                     sw，ss，ws，wsuper， ssuper


        """

        batch_size = g.number_of_nodes(ntype='supernode')
        sent_emb = g.ndata['h']['sentence']
        score = score.reshape(-1, 1)

        g.nodes['sentence'].data['h'] = torch.mul(score, sent_emb)
        g.nodes['word'].data['z'] = self.w_fc(g.ndata['h']['word'])
        g.nodes['sentence'].data['z'] = self.s_fc(g.ndata['h']['sentence'])
        g.nodes['supernode'].data['z'] = self.super_fc(g.ndata['h']['supernode'])
        g.nodes['word'].data['u'] = self.w_att_fc(g.ndata['h']['word'])
        g.nodes['sentence'].data['u'] = self.s_att_fc(g.ndata['h']['sentence'])

        # 更新单词节点表示: 获取邻居节点（句子节点）表示
        wnodes = [i for i in range(g.num_nodes('word'))]
        g.apply_edges(self.sw_attention, etype='sw')  # 给边上添加注意力分数，src(sentence)->dst(word)的权重值
        g['sw'].pull(wnodes, self.sw_message, self.sw_reduce, etype='sw')  # 针对sw边获取权重加和操作
        w_h = F.sigmoid(self.w_h_fc(
            torch.cat([g.ndata['h']['word'], g.ndata['n']['word']],
                      dim=1)))  # n是s_output_dim和h是w_input_dim的维度不一样 [word_node_num , (s_output_dim + w_input_dim)]

        # 更新句子节点表示，获取邻居节点（包括句子节点和单词节点）表示
        snodes = [i for i in range(g.num_nodes('sentence'))]
        g.apply_edges(self.ss_attention, etype='ss')
        g.apply_edges(self.ws_attention, etype='ws')
        g['ss'].pull(snodes, self.ss_message, self.ss_reduce, etype='ss')
        g['ws'].pull(snodes, self.ws_message, self.ws_reduce, etype='ws')
        s_h = F.sigmoid(self.s_h_fc(
            torch.cat([g.ndata['ns']['sentence'], g.ndata['nw']['sentence'], g.ndata['h']['sentence']],
                      dim=1)))

        # 更新supernode
        g.apply_edges(self.ssuper_attention, etype='ssuper')
        g.apply_edges(self.wsuper_attention, etype='wsuper')
        g['ssuper'].pull(0, self.ssuper_message, self.ssuper_reduce, etype='ssuper')
        g['wsuper'].pull(0, self.wsuper_message, self.wsuper_reduce, etype='wsuper')

        super_h = F.sigmoid(self.super_h_fc(
            torch.cat([g.ndata['ns']['supernode'], g.ndata['nw']['supernode'], g.ndata['h']['supernode']], dim=1)))

        # 计算句子节点相对于supernode的重要性分数
        supernode_h = g.ndata['h']['supernode']
        sentences_h = g.ndata['h']['sentence']
        sentences_h = sentences_h.reshape(batch_size, -1, sentences_h.shape[1])
        supernode_h = supernode_h.unsqueeze(dim=2)
        new_scores = torch.softmax(
            (torch.bmm(sentences_h, supernode_h).squeeze() / torch.norm(
                supernode_h.float(), dim=1)).float(), dim=1)

        g.nodes['word'].data['h'] = w_h
        g.nodes['sentence'].data['h'] = s_h
        g.nodes['supernode'].data['h'] = super_h

        # 去除本层聚合的邻居信息
        g.ndata.pop('n')
        g.ndata.pop('nw')
        g.ndata.pop('ns')
        # TODO: return word node representation and sentence node representation and scores
        return new_scores, g.nodes['sentence'].data['h'], g.nodes['supernode'].data['h']

    # sent -> word
    # 聚合两端点信息，然后全联接得到标量权重分数
    def sw_attention(self, edges):
        u = torch.cat([edges.src['z'], edges.dst['z']],
                      dim=1)  # [edge_num, s_output_dim + w_output_dim]
        att = F.leaky_relu(self.s_w_attention_fc(u))  # [edge_num, 1]

        return {'att': att}

    def sw_message(self, edges):
        return {"att": edges.data['att'], "u": edges.src['u']}  # 返回邻居节点的隐状态，和对应连接边上的权重值

    def sw_reduce(self, nodes):
        att = F.softmax(nodes.mailbox['att'], dim=1)
        h = torch.sigmoid(torch.sum(att * nodes.mailbox['u'], dim=1))
        return {'n': h}

    # sent -> sent
    def ss_attention(self, edges):
        u = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        att = F.leaky_relu(self.s_s_attention_fc(u))
        return {'att': att}

    def ss_message(self, edges):
        return {"att": edges.data['att'], "u": edges.src['u']}  # 返回邻居节点的隐状态，和对应连接边上的权重值

    def ss_reduce(self, nodes):
        att = F.softmax(nodes.mailbox['att'], dim=1)
        h = torch.sigmoid(torch.sum(att * nodes.mailbox['u'], dim=1))
        return {'ns': h}

    # word -> sent
    def ws_attention(self, edges):
        u = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        att = F.leaky_relu(self.w_s_attention_fc(u))
        return {'att': att}

    def ws_message(self, edges):
        return {"att": edges.data['att'], "u": edges.src['u']}  # 返回邻居节点的隐状态，和对应连接边上的权重值

    def ws_reduce(self, nodes):
        att = F.softmax(nodes.mailbox['att'], dim=1)
        h = torch.sigmoid(torch.sum(att * nodes.mailbox['u'], dim=1))
        return {'nw': h}

    # word -> supernode
    def wsuper_attention(self, edges):
        u = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        att = F.leaky_relu(self.w_super_attention_fc(u))
        return {'att': att}

    def wsuper_message(self, edges):
        return {"att": edges.data['att'], "u": edges.src['u']}  # 返回邻居节点的隐状态，和对应连接边上的权重值

    def wsuper_reduce(self, nodes):
        att = F.softmax(nodes.mailbox['att'], dim=1)
        h = torch.sigmoid(torch.sum(att * nodes.mailbox['u'], dim=1))
        return {'nw': h}

    # sent -> supernode
    def ssuper_attention(self, edges):
        u = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        att = F.leaky_relu(self.s_super_attention_fc(u))
        return {'att': att}

    def ssuper_message(self, edges):
        return {"att": edges.data['att'], "u": edges.src['u']}  # 返回邻居节点的隐状态，和对应连接边上的权重值

    def ssuper_reduce(self, nodes):
        att = F.softmax(nodes.mailbox['att'], dim=1)
        h = torch.sigmoid(torch.sum(att * nodes.mailbox['u'], dim=1))
        return {'ns': h}


# TODO: 句子对抽取，NNPM来更新边的特征（二分类），边的权重信息由AA.t()计算赋值得到，节点初始表示由之前GAT得到的句子表示
class NNPM(torch.nn.Module):
    def __init__(self, s_input_dim, s_output_dim):
        super(NNPM, self).__init__()
        self.edge_feat = torch.nn.Linear(s_input_dim * 2, s_output_dim)

    def forward(self, batch_g):  # g是批图，sent_embed是[batch_size, sent_num, dim]
        # batch_g, idx = map(list, zip(*batch_g))
        batch_g.apply_edges(self.edge_cls)
        return batch_g.edata.pop('res')  # [batch_size * edge_num, 2] ,edges_num = sent_num * sent_num

    def edge_cls(self, edges):
        feat = self.edge_feat(edges.data['w'] * torch.cat([edges.src['h'], edges.dst['h']], dim=1))
        feat = F.softmax(F.sigmoid(feat), dim=1)
        edges_w = torch.cat((1 - edges.data['w'], edges.data['w']), dim=1)
        feat = torch.mul(feat, edges_w)
        # TODO: softmax???
        # feat = torch.softmax(feat, dim=1)
        return {'res': feat}


class LayerNorm(nn.Module):
    """Construct a layernorm module  : Graph-wise Normalization"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GraphNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, is_node=True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        self.is_node = is_node

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('normalization_gamma', None)
            self.register_parameter('normalization_beta', None)

    def norm(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.std(dim=0, keepdim=True)
        x = (x - mean) / (var + self.eps)
        return x

    def forward(self, g, x):
        graph_size = g.batch_num_nodes if self.is_node else g.batch_num_edges
        x_list = torch.split(x, graph_size)
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = torch.cat(norm_list, 0)

        if self.affine:
            return self.gamma * norm_x + self.beta
        else:
            return norm_x


# Adjance norm for node
class AdjaNodeNormLayer(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(AdjaNodeNormLayer, self).__init__()
        self.eps = eps
        self.affine = affine
        self.num_features = num_features

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def message_func(self, edges):
        return {"h": edges.src["norm_h"]}

    def reduce_func(self, nodes):
        dst_h = nodes.mailbox['h']
        src_h = nodes.data['h']

        h = torch.cat([dst_h, src_h.unsqueeze(1)], 1)
        mean = torch.mean(h, dim=(1, 2))
        var = torch.std(h, dim=(1, 2))

        mean = mean.unsqueeze(1).expand_as(src_h)
        var = var.unsqueeze(1).expand_as(src_h)
        return {"norm_mean": mean, "norm_var": var}

    def forward(self, g, h):
        g.ndata["norm_h"] = h
        g.update_all(self.message_func, self.reduce_func)

        mean = g.ndata['norm_mean']
        var = g.ndata['norm_var']

        norm_h = (h - mean) / (var + self.eps)

        if self.affine:
            return self.gamma * norm_h + self.beta
        else:
            return norm_h


# Adjance norm for edge
class AdjaEdgeNormLayer(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(AdjaEdgeNormLayer, self).__init__()
        self.eps = eps
        self.affine = affine
        self.num_features = num_features

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def message_func(self, edges):
        return {"e": edges.data['norm_e']}

    def reduce_func(self, nodes):
        e = nodes.mailbox['e']
        mean = torch.mean(e, dim=(1, 2))
        var = torch.std(e, dim=(1, 2))

        mean = mean.unsqueeze(1).expand_as(e[:, 0, :])
        var = var.unsqueeze(1).expand_as(e[:, 0, :])
        return {"norm_mean": mean, "norm_var": var}

    def apply_edges(self, edges):
        mean = edges.dst['norm_mean']
        var = edges.dst['norm_var']
        return {"edge_mean": mean, "edge_var": var}

    def forward(self, g, e):
        g.edata['norm_e'] = e
        g.update_all(self.message_func, self.reduce_func)
        g.apply_edges(self.apply_edges)
        mean = g.edata['edge_mean']
        var = g.edata['edge_var']

        norm_e = (e - mean) / (var + self.eps)
        if self.affine:
            return self.gamma * norm_e + self.beta
        else:
            return norm_e


class SublayerConnection(nn.Module):
    """
        A residual connection followed by a layer norm.
        Note for code simplicity the norm is first as opposed to last.
        """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))
