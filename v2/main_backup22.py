import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import dgl
from dgl.data.utils import load_graphs
import gensim
import time
import logging
import os
import sys
import traceback
from rouge import Rouge
from util import logger, f_logger, save_model, read_txt_data, read_json_data, preprocess, extract_keyword, \
    cal_word_freq, \
    get_word_id, get_sent_id, padding_sent_matrix_id, get_vocab_embedding, PAD_TOKEN, UNKNOWN_TOKEN

gpu = 2

import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
import math

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)


setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]


class DocGraph(Dataset):
    # 怎样获取批处理中文档的最大长度
    def __init__(self, data_path, vocab, filter_word, max_keyword_num):
        """
        :param data_path: doc data path
        :param vocab: vocab
        :param filter_word: stop word txt
        :param max_sent_num: max sentence number of a document
        :param max_sent_len: max length of a sentence
        :param max_keyword_num: extract top k keywords

        """
        self.docs = read_json_data(data_path)
        self.vocab = vocab
        self.filter_words = filter_word
        self.max_keyword_num = max_keyword_num
        self.doc_item = []

    def get_doc_item(self, index):
        return self.docs[index]

    def __getitem__(self, index):
        # print("---doc index-----")
        # print(index)
        start_time = time.time()
        doc = self.docs[index]
        text = doc["text"]
        summary = doc.get("summary", [])
        # summary = doc.set_default("summary", [])
        label = doc["label"]

        processed_text, entitties = preprocess(text, self.filter_words)
        keywords = extract_keyword(" ".join(processed_text), steps=20, top_k=self.max_keyword_num,
                                   allow_pos=['NOUN', 'PROUN'])
        words = entitties.union(keywords)
        # 去除低频词,低频词的频率参数, TODO: 去除数字
        word_freq = cal_word_freq(" ".join(processed_text), words)
        res_words = set()
        for word in list(words):
            # if word_freq.get(word, 0) < 1:  # 去除低频词
            #     words.remove(word)
            # else:
            ww = word.split()
            res_words.update(ww)
        # 统计句子最大长度（单词个数），然后padding到固定长度
        gra = self.__create_graph__(res_words, processed_text, label)
        logger.info('get item time: {:5.2f}s'.format(time.time() - start_time))
        return gra, index

    def __len__(self):
        return len(self.docs)

    def __create_graph__(self, words, text, label):
        """
        TODO: 怎样去构建一张图，
        word node: keyword and entity
        sentence node: sentences of document
        doc node(super node): link to sentence node and word node

        :return:
        """
        # TODO: 使用异质网络图
        # 单词不在字典中，padding的字符
        # TODO: 添加单词节点（单词在字典中的序号，本身在图中的序号），句子节点（句子在文中的序号，本身在图中的序号），supernode（额外的一个节点）
        # TODO: 添加单词-句子边（如果单词出现在句子中，则添加边），添加句子-句子边（如果两个句子中出现共同的words中的word，则添加边）
        # TODO: 添加super node到其他所有节点的边 都是无向边
        logging.info('create graph...')
        vocab_word_id = get_word_id(self.vocab)  # word在字典中的id
        sent_matrix_id = get_sent_id(text,
                                     vocab_word_id)  # 矩阵形式， sent_num * sent_len, 矩阵每行存储句子中单词的在词典的id,注意sent_len是不固定的

        UNKNOWN_TOKEN_ID = vocab_word_id[UNKNOWN_TOKEN]
        PAD_TOKEN_ID = vocab_word_id[PAD_TOKEN]

        wid_nid = {}
        nid_wid = {}
        # words中抽取到有些是短语（实体）,那么就把短语拆分成单词再看是否存在于字典中

        for nid, word in enumerate(words):
            wid = vocab_word_id.get(word, UNKNOWN_TOKEN_ID)
            nid_wid[nid] = wid
            wid_nid[wid] = nid

        w_nodes = len(nid_wid)
        s_nodes = len(sent_matrix_id)

        sw_edges = [[], []]
        ss_edges = [[], []]
        wsuper_edges = [[], []]
        ssuper_edges = [[], []]
        graph_data = {}

        # TODO: 建立word-word 无向边， 两个单词在同一个句子中共现则添加连线

        # 建立 word-sent 无向边， 只要单词出现在该句子中就建立联系
        for s_nid, s_row_ids in enumerate(sent_matrix_id):  # 句子id, 句子中包含词的在词典中的id
            # for w_id in [vocab_word_id[word] for word in words]: # KeyError: 'today al - musmari'
            for w_id in list(nid_wid.values()):
                if w_id in s_row_ids and w_id != UNKNOWN_TOKEN_ID:
                    sw_edges[0].append(s_nid)
                    sw_edges[1].append(wid_nid[w_id])

        # 建立sent-sent 无向边，只要两个句子有公有单词
        for i in range(len(sent_matrix_id)):
            for j in range(i + 1, len(sent_matrix_id)):
                sent_i = set(sent_matrix_id[i])
                sent_j = set(sent_matrix_id[j])
                intersect_word_id = sent_i.intersection(sent_j)
                if intersect_word_id != {UNKNOWN_TOKEN_ID} and len(intersect_word_id) > 0:
                    ss_edges[0].append(i)
                    ss_edges[1].append(j)

        # 建立super-word和super-sent无向边,一张图中只有一个super node
        for w_nid in range(w_nodes):
            wsuper_edges[0].append(w_nid)
            wsuper_edges[1].append(0)

        for s_nid in range(s_nodes):
            ssuper_edges[0].append(s_nid)
            ssuper_edges[1].append(0)

        graph_data[('sentence', 'sw', 'word')] = (torch.LongTensor(sw_edges[0]), torch.LongTensor(sw_edges[1]))
        graph_data[('word', 'ws', 'sentence')] = (torch.LongTensor(sw_edges[1]), torch.LongTensor(sw_edges[0]))
        graph_data[('sentence', 'ss', "sentence")] = (torch.LongTensor(ss_edges[0]), torch.LongTensor(ss_edges[1]))
        graph_data[('sentence', 'ss', "sentence")] = (torch.LongTensor(ss_edges[1]), torch.LongTensor(ss_edges[0]))
        graph_data[('word', 'wsuper', 'supernode')] = (
            torch.LongTensor(wsuper_edges[0]), torch.LongTensor(wsuper_edges[1]))
        # graph_data[('supernode', 'superw', 'word')] = (torch.tensor(wsuper_edges[1]), torch.tensor(wsuper_edges[0]))
        graph_data[('sentence', 'ssuper', 'supernode')] = (
            torch.LongTensor(ssuper_edges[0]), torch.LongTensor(ssuper_edges[1]))
        # graph_data[('supernode', 'supers', 'sentence')] = (torch.tensor(ssuper_edges[1]), torch.tensor(ssuper_edges[0]))

        g = dgl.heterograph(graph_data)
        # TODO: UNKNOWN_TOKEN存在问题，如果存在多个UNKNOWN_TOKEN那多个句子节点都指向同一个UNKNOWN_TOKEN节点
        g.nodes['word'].data['wid'] = torch.LongTensor(list(nid_wid.values()))

        # TODO: 需要给一篇文档中的每个句子固定到一个统一长度
        pad_sent_matrix_id = padding_sent_matrix_id(sent_matrix_id, PAD_TOKEN_ID)

        g.nodes['sentence'].data['wids'] = torch.LongTensor(pad_sent_matrix_id)  # 每个句子填充成该文档中最长句子长度，用PAD_TOKEN填充
        g.nodes['sentence'].data['position'] = torch.arange(0, len(sent_matrix_id)).view(-1, 1).long()
        g.nodes['sentence'].data['label'] = torch.LongTensor(s_nodes * [label])  # sentence_num * labels_num

        # 异质网络图添加句子节点特征（包含的单词id和自身句子在文中的顺序），句子对 标签, 句子表示
        # 异质网络图添加单词节点特征 ， 单词表示？节点在词典中的id

        return g


""""
句子长度padding：？？？句子编码使用？？？使用RNN编码时使用

句子节点数目padding：？？？图的节点数使用？？？
单词节点数目padding：padding一些[PAD]无用词，即一些孤立点
1个supernode

句子的最大单词数固定
文档最大句子数固定
这样整个图的规模节点数就固定住

"""


# https://zhuanlan.zhihu.com/p/60129684

class PadBatchGraph:
    def __init__(self, vocab, dataset):
        self.vocab = vocab
        self.vocab_word_id = get_word_id(self.vocab)
        self.labels = self.get_labels(dataset)

    def get_labels(self, docs):
        labels = [doc['label'] for doc in docs]
        return labels

    def pad_colllate(self, batch_g):
        # batch_g = dgl.unbatch(batch_g)
        start_time = time.time()
        batch_g, indexes = map(list, zip(*batch_g))

        max_snode_num = max([g.number_of_nodes(ntype="sentence") for g in batch_g])  # max sent node num in batch graphs
        max_wnode_num = max([g.number_of_nodes(ntype="word") for g in batch_g])  # max word node num in batch graphs
        max_sent_len = max([len(g.ndata['wids']['sentence'][0]) for g in batch_g])
        batch_g_lcs = []
        # padding: 给batch中每个g添加句子节点和单词节点，知道当前g中单词节点的个数和句子节点的个数
        for index, g in zip(indexes, batch_g):
            wnode_num = g.number_of_nodes(ntype="word")
            snode_num = g.number_of_nodes(ntype="sentence")

            # 新添加的单词节点
            g.add_nodes(max_wnode_num - wnode_num, ntype="word")
            # 新添加单词节点添加属性 单词ID
            for ix in range(wnode_num, max_wnode_num):
                g.ndata['wid']['word'][ix] = self.vocab_word_id[PAD_TOKEN]

            raw_labels = self.labels[index]
            sent_len = len(g.ndata['wids']['sentence'][0])
            # 句子长度补齐
            sent_matrix_id = g.ndata['wids']['sentence'].numpy().tolist()
            for row in sent_matrix_id:
                row.extend([self.vocab_word_id[PAD_TOKEN]] * (max_sent_len - sent_len))
            # 添加句子节点，并对新添加的句子节点
            g.add_nodes(max_snode_num - snode_num, ntype="sentence")
            sent_matrix_id.extend([[self.vocab_word_id[PAD_TOKEN]] * max_sent_len] * (max_snode_num - snode_num))
            g.nodes['sentence'].data['wids'] = torch.LongTensor(sent_matrix_id)
            # 句子节点添加属性 位置，包含的单词ID序列
            for ixx in range(snode_num, max_snode_num):
                g.ndata['position']['sentence'][ixx] = torch.LongTensor([ixx])

            label_matrix = torch.zeros([max_snode_num, 1])
            # 构建标签，[max_sent_num, max_sent_num]
            for idx in range(len(raw_labels)):
                label_matrix[raw_labels[idx]][0] = 1

            g.nodes['sentence'].data['label'] = label_matrix

            # 创建全联接sent2sent图
            u = [id for id in range(max_snode_num) for _ in range(max_snode_num)]
            v = [id for id in range(max_snode_num)] * max_snode_num
            g_cls = dgl.graph((u, v))
            batch_g_lcs.append(g_cls)
        logger.info('pad batch time: {:5.2f}'.format(time.time() - start_time))
        return dgl.batch(batch_g), dgl.batch(batch_g_lcs), indexes

    def __call__(self, batch):
        return self.pad_colllate(batch)


# TODO: 引入位置信息，句子在原文中的位置信息，单词在句子的位置信息

"""
attention_score对应每个句子表示的相对于全文表示的重要性分数，该分数去乘每个句子表示后再送入到下一层gcn中

"""


class GraphConvolutionLayer(torch.nn.Module):

    def __init__(self, w_input_dim, s_input_dim, w_output_dim, s_output_dim):  # supernode_output dim和sentence dim相同
        super(GraphConvolutionLayer, self).__init__()
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
        new_scores = (torch.bmm(sentences_h, supernode_h).squeeze() / torch.norm(supernode_h.float(), dim=1)).float()

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
    def __init__(self, s_input_dim, s_output_dim, prior):
        super(NNPM, self).__init__()
        self.edge_feat = torch.nn.Linear(s_input_dim * 2, s_output_dim)
        self.edge_feat.bias.data.fill_(-math.log((1 - prior) / prior))

    def forward(self, batch_g):  # g是批图，sent_embed是[batch_size, sent_num, dim]
        # batch_g, idx = map(list, zip(*batch_g))
        batch_g.apply_edges(self.edge_cls)
        return batch_g.edata.pop('res')  # [batch_size * edge_num, 2] ,edges_num = sent_num * sent_num

    def edge_cls(self, edges):
        feat = F.sigmoid(self.edge_feat(edges.data['w'] * torch.cat([edges.src['h'], edges.dst['h']], dim=1)))

        return {'res': feat}
        # weight = torch.cat((1 - edges.data['w'], edges.data['w']), dim=1)
        # return {'res': torch.mul(feat, weight)}

        # return {'res': self.edge_feat(torch.cat([edges.src['h'], edges.dst['h']], dim=1))}


# TODO: 每一层搞一个多头注意力,拼接或者取平均  节点向量的赋值在最顶层模型给出
class GraphConvolutionNet(torch.nn.Module):
    def __init__(self, w_input_dim, w_hidden_dim, w_output_dim, s_input_dim, s_hidden_dim, s_output_dim, prior,
                 embedded):
        super(GraphConvolutionNet, self).__init__()
        # self.prior = prior
        self.layers = torch.nn.ModuleList()
        self.layer_num = len(w_hidden_dim)
        i = 0
        self.layers.append(GraphConvolutionLayer(w_input_dim, s_input_dim, w_hidden_dim[i], s_hidden_dim[i]))
        for i in range(1, self.layer_num):
            # w_input_dim, s_input_dim, w_output_dim, s_output_dim
            self.layers.append(
                GraphConvolutionLayer(w_hidden_dim[i - 1], s_hidden_dim[i - 1], w_hidden_dim[i], s_hidden_dim[i]))
        self.layers.append(GraphConvolutionLayer(w_hidden_dim[i], s_hidden_dim[i], w_output_dim, s_output_dim))
        self.embedded = embedded
        self.cls_layers = NNPM(s_output_dim, 2, prior)
        self.dropout = nn.Dropout(0.2)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=s_input_dim, nhead=6)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)

        self.cls = torch.nn.Linear(s_input_dim, 2)
        self.cls.bias.data.fill_(-math.log((1 - prior) / prior))

    # 获取句子中单词在词典中的id，然后获取id对应的单词，然后根据单词从嵌入矩阵中获取对应的编码
    def forward(self, g, score, g_cls):
        # TODO:  单词句子编码，supernode编码
        set_embedding_time = time.time()
        batch_size = g.number_of_nodes(ntype='supernode')
        wids = g.ndata['wid']['word'].cuda(gpu)

        # set g embedding
        g.nodes['word'].data['h'] = self.embedded(wids)
        sentences_wids = g.ndata['wids']['sentence'].cuda(gpu)
        g.nodes['sentence'].data['h'] = torch.mean(self.embedded(sentences_wids), dim=1)
        sent_emb = g.ndata['h']['sentence']
        sent_emb = sent_emb.reshape(batch_size, -1,
                                    sent_emb.shape[1])  # [258, 300] => [6,43,300]
        # sent_emb = self.transformer_encoder(sent_emb)

        score = torch.unsqueeze(score, dim=1).cuda(gpu)  # [6, 43] => [6, 1, 43]
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

        s_cls = F.sigmoid(self.cls(self.dropout(s_h)))  # [batch_size*sent_num, 2]

        sent_pair = None
        """
        
        # set g_cls embedding
        setcls_embedding_time = time.time()
        # [batch_size * sent_num, dim] => [batch_size, sent_num, dim]
        s_h_reshape = s_h.reshape(batch_size, -1, s_h.shape[1])
        # [batch_size,sent_num,1] * [batch_size,1,sent_num] => [batch_size,sent_num,sent_num]
        weight = torch.bmm(new_score.unsqueeze(dim=2), new_score.unsqueeze(dim=1))
        # (weight)[batch_size,sent_num,sent_num] * (s_ matrix)[batch_size,sent_num,sent_num] => [batch_size,sent_num,sent_num]
        sent_pair_weight = 1 + torch.sigmoid(torch.mul(weight, torch.bmm(s_h_reshape, s_h_reshape.permute(0, 2, 1))))

        g_cls.ndata['h'] = s_h
        g_cls.edata['w'] = sent_pair_weight.reshape(-1, 1)  # [batch_size * sent_num * sent_num, 1]
        logger.info('set on g_cls time: {:5.2f}'.format(time.time() - setcls_embedding_time))

        # calculate on g_cls
        calongcls_embedding_time = time.time()
        sent_pair = self.cls_layers.forward(g_cls)  # [batch_size* sent_num * sent_num, 2]
        g_cls.ndata.pop('h')
        g_cls.edata.pop('w')
        logger.info('calculate on g_cls time: {:5.2f}'.format(time.time() - calongcls_embedding_time))
        """
        return new_score, s_cls, super_h, sent_pair


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predict, labels):
        """"
        :param predict: [batch_size*sent_num*sent_num, 2]
        :param labels: [batch_size*sent_num*sent_num]
        """
        pt = torch.gather(predict, dim=1, index=labels)
        print("label \n", labels.shape)
        pos = torch.full(labels.shape, 0.2, device=torch.device("cuda:" + str(gpu)))
        neg = torch.full(labels.shape, 0.8, device=torch.device("cuda:" + str(gpu)))

        self.alpha = torch.where(labels == 0, neg, pos)
        return torch.sum(-self.alpha * torch.pow((1 - pt), self.gamma) * pt.log())


class Loss(torch.nn.Module):
    def __init__(self, focal_loss, batch_size, prior, loss_lambda):
        super().__init__()
        self.focal_loss = focal_loss
        self.batch_size = batch_size
        self.T = 100
        self.prior = prior  # 只要正样本概率值超过阈值prior就选中
        self.k = 1 / self.prior
        self.relu = nn.ReLU()
        self.loss_lambda = loss_lambda

    def forward(self, score, sent_pair, s_h, super_h, labels, sent_num):
        """
        :param score: [batch_size, sent_num] 可以做成权重矩阵按位运算到sent_pair上
        :param sent_pair: [batch_size*sent_num*sent_num, 2]
        :param s_h: [batch_size*sent_num, dim]
        :param super_h: [batch_size, dim]
        :param labels: [batch_size*sent_num*sent_num]
        :param sent_sum: num of sentence of each batch
        :return:
        """
        # TODO: 加上s_super,s_s值后会出现sent_pair中nan值 ： 学习率太高，loss函数问题，
        #  对于回归问题，分母为0，target本身应该能够被loss函数计算，比如sigmoid激活函数的target应该大于0
        # TODO: 将得到的每个句子的二分类概率值进行relu化，即只要正样本大于prior就被选中，而非大于0.5
        # sent_pair[:, 1] = self.k * self.relu(sent_pair[:, 1] - self.prior) + self.prior
        # sent_pair[:, 0] = 1 - sent_pair[:, 1]
        # sent_pair = F.softmax(sent_pair, dim=1)
        focal_loss = self.focal_loss(s_h, labels.long())
        logger.info("[INFO] focal loss")
        logger.info(focal_loss)

        """"
        
        sent_pair = sent_pair.reshape(self.batch_size, sent_num, sent_num, 2)
        s_h = s_h.reshape(self.batch_size, -1, s_h.shape[1])
        super_h = F.normalize(super_h, dim=1).unsqueeze(dim=2)

        # sent和sent之间相似度尽可能的小，交集小
        s_norm_h = F.normalize(s_h, dim=2)
        sim_s_s_matrix = torch.bmm(s_norm_h, s_norm_h.permute(0, 2, 1))  # [batch_size, sent_num, sent_num]
        mask = F.softmax(self.T * sent_pair, dim=1)[:, :, :, 1:].reshape(self.batch_size, sent_num,
                                                                         sent_num)  # [batch_size, sent_num, sent_num]
        sim_s_s = torch.mean(torch.mul(mask, sim_s_s_matrix))
        # sent和super之间的相似度尽可能大
        sim_s_super_matrix = torch.bmm(s_norm_h, super_h)  # [batch_size, sent_sum,1]
        mask2 = torch.sum(mask, dim=2).unsqueeze(dim=1)  # [batch_size, 1, sent_num]
        sim_s_super = torch.sum(torch.bmm(mask2, sim_s_super_matrix)) / torch.sum(mask)  # [batch_size, 1, 1] = scalar
        # TODO:相似度出现负数，因为余弦计算公式的分子出现负数，即向量表示中有负数，可以使用sigmoid函数来表示到正数范围
        logger.info("sim_s_super")
        logger.info(sim_s_super)
        logger.info("sim_s_s")
        logger.info(sim_s_s)
        # return focal_loss + torch.exp(-1 * sim_s_super) + sim_s_s
        return focal_loss + (1 - self.loss_lambda) * torch.log(
            (sim_s_super + 1 + 1e-5) / 2) + self.loss_lambda * torch.exp(
            sim_s_s + 3)
        """
        return focal_loss


class DatasetGraph(Dataset):
    def __init__(self, dataset_graphs, dataset):
        self.dataset_graphs = dataset_graphs
        self.docs = dataset

    def __getitem__(self, index):
        return self.dataset_graphs[index], index

    def __len__(self):
        return len(self.dataset_graphs)

    def get_doc_item(self, index):
        return self.docs[index]


def validate(hps, model, validate_loader, validate_dataset, best_loss, best_F, non_descent_cnt, saveNo):
    f_logger.info("[info] Starting validate for this model...")
    eval_dir = os.path.join("./model", "eval")
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    model.eval()
    iter_start_time = time.time()

    # 计算在验证集或测试集上的loss
    tester = Tester(hps)
    with torch.no_grad():  # 不需要进行梯度计算
        for i, (G, G_cls, index) in enumerate(validate_loader):
            G = G.to(torch.device("cuda:" + str(hps.gpu)))
            G_cls = G_cls.to(torch.device("cuda:" + str(hps.gpu)))
            tester.validate(model, G, G_cls, validate_dataset, index)
        tester.calc()
    logger.info(tester.candidate)
    running_avg_loss = tester.running_avg_loss
    f_logger.info('[INFO] End of valid | time: {:5.2f}s | valid loss {:5.4f} | '.format((time.time() - iter_start_time),
                                                                                        float(running_avg_loss)))
    # 经过模型分类后解码后，出现句子分类都是0的情况
    if len(tester.candidate) == 0 or len(tester.gold) == 0:
        f_logger.error("During testing, no candidate summary sentence is selected")
        return best_loss, best_F, non_descent_cnt, saveNo

    # 计算抽取的句子与gold summary的rouge值
    rouge = Rouge()
    try:
        scores_all = rouge.get_scores(tester.candidate, tester.gold, avg=True)
    except Exception as e:
        traceback.print_exc()
        f_logger.error(e)
        return best_loss, best_F, non_descent_cnt, saveNo

    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
        scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
          + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
              scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
          + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
              scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    f_logger.info(res)
    logger.info(res)

    if not best_loss or running_avg_loss < best_loss:  # 第一次或者loss下降的情况
        bestmodel_save_path = os.path.join(eval_dir, "bestmodel_%d" % (saveNo % 3))
        if best_loss:  # 不为空
            f_logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is %.6f, Saving to %s',
                float(running_avg_loss), float(best_loss), bestmodel_save_path)
        else:  # 为空
            f_logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is None, Saving to %s',
                float(running_avg_loss), bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        best_loss = running_avg_loss
        non_descent_cnt = 0
        saveNo += 1
    else:
        non_descent_cnt += 1

    if not best_F or best_F < tester.F:
        bestmodel_save_path = os.path.join(eval_dir, "bestFmodel")
        if best_F:
            f_logger.info('[INFO] Found new best model with %.6f F. The original F is %.6f, Saving to %s',
                          float(tester.F),
                          float(best_F), bestmodel_save_path)
        else:
            f_logger.info('[INFO] Found new best model with %.6f F. The original F is None, Saving to %s',
                          float(tester.F),
                          bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        best_F = F

    return best_loss, best_F, non_descent_cnt, saveNo


def train(hps, model, train_loader, validate_loader):
    best_train_loss = None
    best_loss = None
    best_F = None
    non_descent_cnt = 0
    saveNo = 0
    # 多卡运行
    # model = torch.nn.DataParallel(
    #     GraphConvolutionNet(w_input_dim, w_hidden_dim, w_output_dim, s_input_dim, s_hidden_dim, s_output_dim,
    #                         embeded), device_ids=device_ids)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)
    # optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids)

    # bceloss = torch.nn.BCELoss()
    criterion = Loss(FocalLoss(hps.loss_alpha, hps.loss_gamma), hps.batch_size, hps.loss_prior, hps.loss_lambda).cuda(
        hps.gpu)
    # # train
    # # TODO: 每个epoch中记录train loss 和 valid loss的变化, 可视化
    for epoch in range(1, hps.epoch + 1):
        epoch_loss = 0.0
        train_loss = 0.0
        epoch_start_time = time.time()

        for i, (G, G_cls, index) in enumerate(train_loader):
            try:
                iter_start_time = time.time()
                G = G.to(torch.device("cuda:" + str(hps.gpu)))
                G_cls = G_cls.to(torch.device("cuda:" + str(hps.gpu)))
                sent_num = int(G.number_of_nodes(ntype="sentence") / hps.batch_size)
                labels = G.ndata['label']['sentence'].reshape(hps.batch_size * sent_num, -1)

                model.train()
                initiate_scores = torch.tensor(hps.batch_size * [[1 / sent_num] * sent_num]).cuda(hps.gpu)
                # scores: [batch_size,sent_num]  s_h:[batch_size*sent_num, dim]  super_h:[batch_size,dim]
                # sent_pair:[batch_size*sent_num*sent_num, 2]
                forward_start_time = time.time()
                new_score, s_cls, super_h, sent_pair = model.forward(G, initiate_scores, G_cls)
                logger.info('forward time: {:5.2f}s'.format(time.time() - forward_start_time))
                loss_start_time = time.time()
                loss = criterion(new_score, sent_pair, s_cls, super_h, labels, sent_num)
                optimizer.zero_grad()
                loss.backward()
                logger.info('loss time: {:5.2f}s'.format(time.time() - loss_start_time))

                if hps.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), hps.max_grad_norm)
                optimizer.step()

                train_loss += float(loss.data)
                epoch_loss += float(loss.data)

                if i % 1000 == 0:
                    f_logger.info('       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.6f} | '
                                  .format(i, (time.time() - iter_start_time), float(train_loss / 1000)))
                    train_loss = 0.0
            except Exception as e:
                traceback.print_exc()
                f_logger.info(e)
            continue

        epoch_avg_loss = epoch_loss / len(train_loader)
        f_logger.info('   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.6f} | '
                      .format(epoch, (time.time() - epoch_start_time), float(epoch_avg_loss)))

        if not best_train_loss or epoch_avg_loss < best_train_loss:
            save_file_path = os.path.join(hps.train_dir, "bestmodel")
            f_logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s',
                          float(epoch_avg_loss),
                          save_file_path)
            save_model(model, save_file_path)
            best_train_loss = epoch_avg_loss

        elif epoch_avg_loss >= best_train_loss:
            f_logger.error("[Error] training loss does not descent. Stopping supervisor...")
            save_model(model, os.path.join(hps.train_dir, "earlystop"))
            # sys.exit(1)

        # model, validate_loader, best_loss, best_F, non_descent_cnt, saveNo
        # 在每个train dataset epoch上训练完后就去验证集上验证loss值和标签F值和rouge值
        best_loss, best_F, non_descent_cnt, saveNo = validate(hps, model, validate_loader, validate_dataset, best_loss,
                                                              best_F, non_descent_cnt, saveNo)

        if non_descent_cnt >= 3:
            f_logger.error("[Error] val loss does not descent for three times. Stopping supervisor...")
            save_model(model, os.path.join(hps.train_dir, "earlystop"))

    # model.load_state_dict(torch.load('./model/train/bestmodel'))
    # best_loss, best_F, non_descent_cnt, saveNo = validate(hps, model, validate_loader, validate_dataset, best_loss,
    #                                                       best_F, non_descent_cnt, saveNo)


def setup_training(hps, model, train_loader, validate_loader):
    train_dir = os.path.join(hps.save_root, "train")
    if os.path.exists(train_dir) and hps.restore_model != 'None':  # 之前已经生成过了，那么这次直接加载之前生成的模型
        f_logger.info("[INFO] Restoring %s for training...", hps.restore_model)
        bestmodel_file = os.path.join(train_dir, hps.restore_model)
        model.load_state_dict(torch.load(bestmodel_file))
        hps.save_root = hps.save_root + "_reload"

    try:
        train(hps, model, train_loader, validate_loader)
    except KeyboardInterrupt:
        f_logger.error("[ERROR] Caught keyboard interrupt on worker. Stopping supervisor...")
        save_model(model, os.path.join(train_dir, "earlystop"))


"""
Tester：
计算在验证集和测试集上的loss （标签上的），问题：在测试集上需要计算loss了吗
计算在验证集和测试集上抽取的摘要句子和gold summary的rouge-r, rouge-p, rouge-F值
rouge本身的安装
根据model.forward输出得到sent_pair中的标签[batch_size*sent_num*sent_num, 2] 解码 为在文中句子的ID
根据抽取到的ID对应的句子和gold summary计算rouge值

将batch中的每篇candidate_summray和gold_summary计算平均rouge
最终整个验证集和测试集上的rouge值的计算？

计算
"""


class Tester:
    def __init__(self, hps):
        super(Tester, self).__init__()
        self.hps = hps
        self.criterion = Loss(FocalLoss(hps.loss_alpha, hps.loss_gamma), hps.batch_size, hps.loss_prior,
                              hps.loss_lambda).cuda(hps.gpu)
        self.running_avg_loss = 0.0
        self.loss = 0.0
        self.batch = 0
        self.F = 0.0  # 标签的F值，并非摘要句子之间的F值
        self.precision = 0.0
        self.recall = 0.0
        self.candidate = []
        self.gold = []
        self.prior = hps.loss_prior

    # 进行candidate标签的F值的计算，Recall值计算，Precision计算
    # 进行验证集上的loss的计算
    # 对validate_loader中的G和G_ls进行对应映射 gold summary <=> candidate summary  candidate存 gold存
    # 其中candidate summary需要对model.forward出来的sent_pair结果进行解码出哪些句子被抽取

    def validate(self, model, G, G_cls, validate_dataset, indexes):
        """
        :param model: 之前训练得到的模型
        :param G: 表示学习的GAT G
        :param C_cls: 自环全联接的的GCN G_cls
        :param validate_dataset: 验证数据集
        :param index: 当前batch中文档图在验证集中的index索引列表
        :return:
        """
        self.batch += 1
        sent_num = int(G.number_of_nodes(ntype="sentence") / self.hps.batch_size)
        labels = G.ndata['label']['sentence'].reshape(self.hps.batch_size * sent_num * sent_num, -1).cuda(self.hps.gpu)
        initiate_scores = torch.tensor(self.hps.batch_size * [[1 / sent_num] * sent_num]).cuda(self.hps.gpu)
        # scores: [batch_size,sent_num]  s_h:[batch_size*sent_num, dim]  super_h:[batch_size,dim]
        # sent_pair:[batch_size*sent_num*sent_num, 2]
        score, s_cls, super_h, sent_pair = model.forward(G, initiate_scores, G_cls)
        logger.info("sent_pair: ")
        logger.info(sent_pair)
        loss = self.criterion(score, sent_pair, s_cls, super_h, labels, sent_num)
        self.loss += loss

        cls = sent_pair.max(1)[1].reshape(self.hps.batch_size, sent_num, sent_num).cpu()
        # sent_pair = sent_pair.reshape(self.hps.batch_size, sent_num, sent_num, 2).max(3)[1].cpu()

        # 得到映射，填充candidate， gold的句子
        for i in range(self.hps.batch_size):  # 遍历batch中的每个item
            index = indexes[i]
            doc = validate_dataset.get_doc_item(index)
            labels = doc["label"]
            text = doc["text"]
            summary = doc["summary"]
            # self.gold.append([" ".join([text[label] for label in labels])])
            self.gold.append(" ".join(summary))
            candidate_labels = self.decode(cls[i, :, :])
            logger.info("candidate_labels")
            logger.info(len(candidate_labels) == 0)
            candidate_sentence = []
            for c_l in candidate_labels:
                if c_l < len(text):
                    candidate_sentence.append(text[c_l])
            self.candidate.append(" ".join(candidate_sentence))
            logger.info("candidate")
            logger.info(self.candidate)
            # text = " ".join(doc["text"]) # 句子数组拼接成字符串

    def decode(self, sent_pair):
        """
        找到1的位置（即句子对的序号），做成集合即为抽选的句子的id集合
        :param sent_pair: [sent_num, sent_num]的0-1矩阵
        :return:
        """
        candidate = set()
        logger.info("non zero")
        logger.info(torch.nonzero(sent_pair))
        candidate.update(torch.nonzero(sent_pair).cpu().reshape(-1).numpy().tolist())
        return candidate

    # TODO: 矩阵解码算法，考虑一行中有多个出度，或者出现断链的情况
    def calc(self):
        self.running_avg_loss = self.loss / self.batch


# AlTODO: 把词典换了，换成词向量训练出来的那些词构成的词典，保存格式为每行： word embeddding

if __name__ == '__main__':
    class Hps:
        def __init__(self):
            self.train_data_path = "./data/cnndm/train.label.jsonl"
            self.validate_data_path = './data/cnndm/val.label.jsonl'
            self.test_data_path = './data/cnndm/test.label.jsonl'
            self.vocab_path = "./data/vocab.txt"
            self.filter_word_path = "./data/filter_word.txt"
            self.word_embedding_path = "./model/cnndm_word2vec.model"
            self.train_dir = './model/train/'
            self.train_dataset_graphs_path = './data/graph/train_graph.bin'
            self.val_dataset_graphs_path = './data/graph/val_graph.bin'
            self.test_dataset_graphs_path = './data/graph/test_graph.bin'
            self.device_ids = [0, 1]
            # self.gpu = len(self.device_ids)
            self.gpu = 2
            self.max_keyword_num = 20
            self.w_input_dim = 300
            self.s_input_dim = 300
            self.w_output_dim = 128  # 300
            self.s_output_dim = 128  # 200
            self.w_hidden_dim = [64]
            self.s_hidden_dim = [50]
            self.lr = 1e-4
            self.epoch = 20
            self.dropout = 0.1
            self.batch_size = 8
            self.shuffle = False
            self.num_workers = 0
            self.grad_clip = False
            self.max_grad_norm = 1.0

            self.loss_alpha = 0.25
            self.loss_gamma = 2
            self.loss_prior = 0.00001
            self.loss_lambda = 0.5

            self.restore_model = "None"
            self.save_root = "./model"


    hps = Hps()

    # 加载词典，和词典嵌入，和过滤词
    vocab_freq = read_txt_data(hps.vocab_path)
    vocab = [v_f.split()[0] for v_f in vocab_freq]
    word_embedding = gensim.models.Word2Vec.load(hps.word_embedding_path)
    filter_word = read_txt_data(hps.filter_word_path)
    train_docs = read_json_data(hps.train_data_path)
    validate_docs = read_json_data(hps.validate_data_path)
    # 加载训练数据集和验证集
    dataset_start_time = time.time()
    train_dataset = DatasetGraph(load_graphs(hps.train_dataset_graphs_path)[0], train_docs)
    train_loader = DataLoader(train_dataset, batch_size=hps.batch_size, shuffle=hps.shuffle,
                              num_workers=hps.num_workers,
                              collate_fn=PadBatchGraph(vocab, train_docs), drop_last=True)
    logger.info('train dataset start time: {:5.2f}s'.format(time.time() - dataset_start_time))
    del train_dataset
    validate_dataset = DatasetGraph(load_graphs(hps.val_dataset_graphs_path)[0], validate_docs)
    validate_loader = DataLoader(validate_dataset, batch_size=hps.batch_size, shuffle=hps.shuffle,
                                 num_workers=hps.num_workers,
                                 collate_fn=PadBatchGraph(vocab, validate_docs), drop_last=True)

    # 词嵌入句子
    vocab_embedding = get_vocab_embedding(word_embedding, vocab, hps.w_input_dim)
    embeded = torch.nn.Embedding(len(vocab_embedding), hps.w_input_dim, padding_idx=0)
    embeded.weight.data.copy_(torch.Tensor(list(vocab_embedding.values())))
    embeded.weight.requires_grad = True
    embeded = embeded.cuda(hps.gpu)

    # 模型
    model = GraphConvolutionNet(hps.w_input_dim, hps.w_hidden_dim, hps.w_output_dim, hps.s_input_dim, hps.s_hidden_dim,
                                hps.s_output_dim, hps.loss_prior,
                                embeded).cuda(hps.gpu)

    # train
    setup_training(hps, model, train_loader, validate_loader)
