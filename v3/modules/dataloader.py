import time

import dgl
import torch
from dgl.data.utils import save_graphs
from torch.utils.data import Dataset

from util import logger, read_txt_data, read_json_data, preprocess, extract_keyword, \
    cal_word_freq, \
    get_word_id, get_sent_id, padding_sent_matrix_id, PAD_TOKEN, UNKNOWN_TOKEN


class DatasetGraph(Dataset):
    def __init__(self, dataset_graphs, dataset_path):
        self.dataset_graphs = dataset_graphs
        self.docs = read_json_data(dataset_path)

    def __getitem__(self, index):
        return self.dataset_graphs[index], index

    def __len__(self):
        return len(self.dataset_graphs)

    def get_doc_item(self, index):
        return self.docs[index]


"""

将训练集提前结构化好为图形式数据
然后存储起来
后续在训练的时候边省去了创建图的过程，直接读取的数据集即为图


"""

"""
 doc preprocessing: remove stopword / extract keyword / extract entity
 doc encoding / word encoding as initial representation of docgraph node

"""
class PadBatchGraph:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_word_id = get_word_id(self.vocab)

    def pad_colllate(self, batch_g):
        # batch_g = dgl.unbatch(batch_g)
        start_time = time.time()
        batch_g, indexes = map(list, zip(*batch_g))

        max_snode_num = max([g.number_of_nodes(ntype="sentence") for g in batch_g])  # max sent node num in batch graphs
        max_wnode_num = max([g.number_of_nodes(ntype="word") for g in batch_g])  # max word node num in batch graphs
        max_sent_len = max([len(g.ndata['wids']['sentence'][0]) for g in batch_g])
        batch_g_lcs = []
        # padding: 给batch中每个g添加句子节点和单词节点，知道当前g中单词节点的个数和句子节点的个数
        for g in batch_g:
            wnode_num = g.number_of_nodes(ntype="word")
            snode_num = g.number_of_nodes(ntype="sentence")

            # 新添加的单词节点
            g.add_nodes(max_wnode_num - wnode_num, ntype="word")
            # 新添加单词节点添加属性 单词ID
            for ix in range(wnode_num, max_wnode_num):
                g.ndata['wid']['word'][ix] = self.vocab_word_id[PAD_TOKEN]

            raw_labels = g.ndata['label']['sentence'][0]
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

            label_matrix = torch.zeros([max_snode_num, max_snode_num])
            # 构建标签，[max_sent_num, max_sent_num]
            for idx in range(len(raw_labels) - 1):
                label_matrix[raw_labels[idx]][raw_labels[idx + 1]] = 1
            if len(raw_labels) > 0:
                label_matrix[raw_labels[-1]][raw_labels[-1]] = 1
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


class DocGraph:
    def __init__(self, vocab_path, filter_word_path, max_keyword_num):
        vocab_freq = read_txt_data(vocab_path)
        self.vocab = [v_f.split()[0] for v_f in vocab_freq]
        self.filter_words = read_txt_data(filter_word_path)
        self.max_keyword_num = max_keyword_num
        self.doc_item = []

    def create_graph(self, dataset_path, save_graph_path):
        docs = read_json_data(dataset_path)
        dataset_graphs = []
        for index in range(len(docs)):
            start_time = time.time()
            doc = docs[index]
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
            dataset_graphs.append(gra)
            if index % 100 == 0:
                logger.info('   index: {:d}|get item graph time: {:5.2f}s'.format(index, time.time() - start_time))

        save_graphs(save_graph_path, dataset_graphs)

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


if __name__ == "__main__":
    train_data_path = "./data/cnndm/train.label.jsonl"
    val_data_path = "./data/cnndm/val.label.jsonl"
    test_data_path = "./data/cnndm/test.label.jsonl"
    vocab_path = "./data/vocab.txt"
    filter_word_path = "./data/filter_word.txt"
    max_keyword_num = 20
    train_save_graphs_path = "./data/graph/train_graph.bin"
    val_save_graphs_path = "./data/graph/val_graph.bin"
    test_save_graphs_path = "./data/graph/test_graph.bin"
    # data_path, vocab_path, filter_word_path, max_keyword_num
    dataset_graphs = DocGraph(vocab_path, filter_word_path, max_keyword_num)
    dataset_graphs.create_graph(val_data_path, val_save_graphs_path)
    dataset_graphs.create_graph(test_data_path, test_save_graphs_path)
    dataset_graphs.create_graph(train_data_path, train_save_graphs_path)
