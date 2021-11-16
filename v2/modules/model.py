import torch
import torch.nn as nn
from Layer import GraphAttentionLayer
from Layer import NNPM
import time
from util import logger

# TODO: 每一层搞一个多头注意力,拼接或者取平均  节点向量的赋值在最顶层模型给出
class GraphAttentionNet(torch.nn.Module):
    def __init__(self, w_input_dim, w_hidden_dim, w_output_dim, s_input_dim, s_hidden_dim, s_output_dim,
                 embedded):
        super(GraphAttentionNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layer_num = len(w_hidden_dim)

        self.layers.append(GraphAttentionLayer(w_input_dim, s_input_dim, w_hidden_dim[0], s_hidden_dim[0]))
        for i in range(self.layer_num - 1):
            # w_input_dim, s_input_dim, w_output_dim, s_output_dim
            self.layers.append(
                GraphAttentionLayer(w_hidden_dim[i], s_hidden_dim[i], w_hidden_dim[i + 1], s_hidden_dim[i + 1]))
        self.layers.append(GraphAttentionLayer(w_hidden_dim[-1], s_hidden_dim[-1], w_output_dim, s_output_dim))
        self.cls_layers = NNPM(s_output_dim, 2)

        self.embedded = embedded
        self.dropout = nn.Dropout(0.2)

    # 获取句子中单词在词典中的id，然后获取id对应的单词，然后根据单词从嵌入矩阵中获取对应的编码
    def forward(self, g, score, g_cls):
        # TODO:  单词句子编码，supernode编码
        set_embedding_time = time.time()
        batch_size = g.number_of_nodes(ntype='supernode')
        wids = g.ndata['wid']['word']

        # set g embedding
        g.nodes['word'].data['h'] = self.embedded(wids)
        sentences_wids = g.ndata['wids']['sentence']
        g.nodes['sentence'].data['h'] = torch.mean(self.embedded(sentences_wids), dim=1)
        sent_emb = g.ndata['h']['sentence']
        sent_emb = sent_emb.reshape(batch_size, -1,
                                    sent_emb.shape[1])  # [258, 300] => [6,43,300]
        score = torch.unsqueeze(score, dim=1)  # [6, 43] => [6, 1, 43]
        g.nodes['supernode'].data['h'] = torch.bmm(score, sent_emb).squeeze(dim=1)
        logger.info('set embedding time: {:5.2f}'.format(time.time() - set_embedding_time))
        # calculate on g
        calong_embedding_time = time.time()
        new_score = score
        s_h = torch.tensor([])
        super_h = torch.tensor([])
        for i, layer in enumerate(self.layers):
            new_score, s_h, super_h = layer(g, new_score)
        logger.info('calculate on g time: {:5.2f}'.format(time.time() - calong_embedding_time))

        # set g_cls embedding
        # [batch_size * sent_num, dim] => [batch_size, sent_num, dim]
        setcls_embedding_time = time.time()
        s_h_reshape = s_h.reshape(batch_size, -1, s_h.shape[1])
        # [batch_size,sent_num,1] * [batch_size,1,sent_num] => [batch_size,sent_num,sent_num]
        weight = torch.bmm(new_score.unsqueeze(dim=2), new_score.unsqueeze(dim=1))
        # (weight)[batch_size,sent_num,sent_num] * (s_ matrix)[batch_size,sent_num,sent_num] => [batch_size,sent_num,sent_num]
        sent_pair = F.normalize(torch.sigmoid(torch.mul(weight, torch.bmm(s_h_reshape, s_h_reshape.permute(0, 2, 1)))),
                                dim=0)
        g_cls.ndata['h'] = s_h
        g_cls.edata['w'] = sent_pair.reshape(-1, 1)  # [batch_size * sent_num * sent_num, 1]
        logger.info('set on g_cls time: {:5.2f}'.format(time.time() - setcls_embedding_time))

        # calculate on g_cls
        calongcls_embedding_time = time.time()
        sent_pair = self.cls_layers.forward(g_cls)  # [batch_size* sent_num * sent_num, 2]
        g_cls.ndata.pop('h')
        g_cls.edata.pop('w')
        logger.info('calculate on g_cls time: {:5.2f}'.format(time.time() - calongcls_embedding_time))
        return new_score, s_h, super_h, sent_pair

