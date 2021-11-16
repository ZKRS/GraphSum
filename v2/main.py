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
from rouge import Rouge
from util import logger
from util import save_model
from util import read_txt_data
from util import read_json_data
from util import preprocess
from util import extract_keyword
from util import cal_word_freq
from util import get_word_id
from util import get_sent_id
from util import padding_sent_matrix_id
from util import get_vocab_embedding
from util import PAD_TOKEN
from util import UNKNOWN_TOKEN
from util import PadBatchGraph
from util import Loss
import traceback
from DatasetGraph import DatasetGraph
from model import GraphAttentionNet
""""
句子长度padding：？？？句子编码使用？？？使用RNN编码时使用

句子节点数目padding：？？？图的节点数使用？？？
单词节点数目padding：padding一些[PAD]无用词，即一些孤立点
1个supernode

句子的最大单词数固定
文档最大句子数固定
这样整个图的规模节点数就固定住

"""
# TODO: 引入位置信息，句子在原文中的位置信息，单词在句子的位置信息

def validate(model, validate_loader, validate_dataset, best_loss, best_F, non_descent_cnt, saveNo, batch_size):
    logger.info("[info] Starting validate for this model...")
    eval_dir = os.path.join("./model", "eval")
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    model.eval()
    iter_start_time = time.time()

    # 计算在验证集或测试集上的loss
    with torch.no_grad():  # 不需要进行梯度计算
        tester = Tester(batch_size)
        for i, (G, G_cls, index) in enumerate(validate_loader):
            G = G.to(torch.device("cuda:2"))
            G_cls = G_cls.to(torch.device("cuda:2"))
            tester.validate(model, G, G_cls, validate_dataset, index)
        tester.calc()

    running_avg_loss = tester.running_avg_loss
    logger.info('[INFO] End of valid | time: {:5.2f}s | valid loss {:5.4f} | '.format((time.time() - iter_start_time),
                                                                                      float(running_avg_loss)))
    # 计算抽取的句子与gold summary的rouge值
    rouge = Rouge()
    scores_all = rouge.get_scores(tester.candidate, tester.gold, avg=True)
    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
        scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
          + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
              scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
          + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
              scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    logger.info(res)

    if best_loss is None or running_avg_loss < best_loss:
        bestmodel_save_path = os.path.join(eval_dir, "bestmodel_%d" % (saveNo % 3))
        if best_loss is not None:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is %.6f, Saving to %s',
                float(running_avg_loss), float(best_loss), bestmodel_save_path)
        else:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is None, Saving to %s',
                float(running_avg_loss), bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        best_loss = running_avg_loss
        non_descent_cnt = 0
        saveNo += 1
    else:
        non_descent_cnt += 1

    if best_F is None or best_F < tester.F:
        bestmodel_save_path = os.path.join(eval_dir, "bestFmodel")
        if best_F is not None:
            logger.info('[INFO] Found new best model with %.6f F. The original F is %.6f, Saving to %s',
                        float(tester.F),
                        float(best_F), bestmodel_save_path)
        else:
            logger.info('[INFO] Found new best model with %.6f F. The original F is None, Saving to %s',
                        float(tester.F),
                        bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        best_F = F

    return best_loss, best_F, non_descent_cnt, saveNo


def train():
    pass

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
    def __init__(self, batch_size):
        super(Tester, self).__init__()
        self.batch_size = batch_size
        self.criterion = Loss(CrossEntropyLoss(), self.batch_size)
        self.running_avg_loss = 0.0
        self.loss = 0.0
        self.batch = 0
        self.F = 0.0  # 标签的F值，并非摘要句子之间的F值
        self.precision = 0.0
        self.recall = 0.0
        self.candidate = []
        self.gold = []

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
        sent_num = int(G.number_of_nodes(ntype="sentence") / self.batch_size)
        labels = G.ndata['label']['sentence'].reshape(self.batch_size * sent_num * sent_num)
        initiate_scores = torch.tensor(self.batch_size * [[1 / sent_num] * sent_num])
        # scores: [batch_size,sent_num]  s_h:[batch_size*sent_num, dim]  super_h:[batch_size,dim]
        # sent_pair:[batch_size*sent_num*sent_num, 2]
        score, s_h, super_h, sent_pair = model.forward(G, initiate_scores, G_cls)
        loss = criterion(score, sent_pair, s_h, super_h, labels, sent_num)
        self.loss += loss
        # G_list = dgl.unbatch(G)
        # G_cls_list = dgl.unbatch(G_cls)

        sent_pair = sent_pair.reshape(self.batch_size, sent_num, sent_num, 2).max(3)[1]

        # 得到映射，填充candidate， gold的句子
        for i in range(self.batch_size):  # 遍历batch中的每个item
            index = indexes[i]
            doc = validate_dataset.get_doc_item(index)
            labels = doc["label"]
            text = doc["text"]
            summary = doc["summary"]
            # self.gold.append([" ".join([text[label] for label in labels])])
            self.gold.append(" ".join(summary))
            candidate_labels = self.decode(sent_pair[i, :, :])
            self.candidate.append(" ".join([text[c_l] for c_l in candidate_labels]))
            # text = " ".join(doc["text"]) # 句子数组拼接成字符串

    def decode(self, sent_pair):
        """
        找到1的位置（即句子对的序号），做成集合即为抽选的句子的id集合
        :param sent_pair: [sent_num, sent_num]的0-1矩阵
        :return:
        """
        candidate = set()
        candidate.update(torch.nonzero(sent_pair).reshape(-1).numpy().tolist())
        return candidate

    # TODO: 矩阵解码算法，考虑一行中有多个出度，或者出现断链的情况
    def calc(self):
        self.running_avg_loss = self.loss / self.batch


# AlTODO: 把词典换了，换成词向量训练出来的那些词构成的词典，保存格式为每行： word embeddding

if __name__ == '__main__':
    train_data_path = "./data/cnndm/train.label.jsonl"
    validate_data_path = './data/cnndm/val.label.jsonl'
    vocab_path = "./data/vocab.txt"
    filter_word_path = "./data/filter_word.txt"
    word_embedding_path = "./model/cnndm_word2vec.model"
    train_dir = './model/train/'
    train_dataset_graphs_path = './data/graph/train_graph.bin'
    val_dataset_graphs_path = './data/graph/val_graph.bin'
    test_dataset_graphs_path = './data/graph/test_graph.bin'
    device_ids = [0, 1, 2, 3]
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    max_keyword_num = 20
    w_input_dim = 300
    s_input_dim = 300
    w_output_dim = 64  # 300
    s_output_dim = 64  # 200
    w_hidden_dim = [128]
    s_hidden_dim = [128]
    lr = 1e-4
    epoch = 20
    dropout = 0.1
    batch_size = 8
    shuffle = False
    num_workers = 8
    grad_clip = False
    max_grad_norm = 1.0

    vocab_freq = read_txt_data(vocab_path)
    vocab = [v_f.split()[0] for v_f in vocab_freq]
    word_embedding = gensim.models.Word2Vec.load(word_embedding_path)
    filter_word = read_txt_data(filter_word_path)

    dataset_start_time = time.time()
    train_dataset = DatasetGraph(load_graphs(train_dataset_graphs_path)[0])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                              collate_fn=PadBatchGraph(vocab), drop_last=True)
    logger.info('train dataset start time: {:5.2f}s'.format(time.time() - dataset_start_time))

    del train_dataset
    validate_dataset = DatasetGraph(load_graphs(val_dataset_graphs_path)[0])
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 collate_fn=PadBatchGraph(vocab),drop_last=True)

    vocab_embedding = get_vocab_embedding(word_embedding, vocab, w_input_dim)
    embeded = torch.nn.Embedding(len(vocab_embedding), w_input_dim, padding_idx=0)
    embeded.weight.data.copy_(torch.Tensor(list(vocab_embedding.values())))
    embeded.weight.requires_grad = True
    emebeded = embeded.cuda(2)

    model = GraphAttentionNet(w_input_dim, w_hidden_dim, w_output_dim, s_input_dim, s_hidden_dim, s_output_dim,
                                embeded).cuda(2)
    # if torch.cuda.device_count() > 1:
    #     print("use ", torch.cuda.device_count(), "GPUs")
    #     model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids)

    # bceloss = torch.nn.BCELoss()
    criterion = Loss(CrossEntropyLoss(), batch_size).cuda(2)

    best_train_loss = None
    best_loss = None
    best_F = None
    non_descent_cnt = 0
    saveNo = 0

    # train
    # TODO: 每个epoch中记录train loss 和 valid loss的变化
    for epoch in range(1, epoch + 1):
        epoch_loss = 0.0
        train_loss = 0.0
        epoch_start_time = time.time()
        for i, (G, G_cls, index) in enumerate(train_loader):
            try:
                iter_start_time = time.time()
                G = G.to(torch.device("cuda:2"))
                G_cls = G_cls.to(torch.device("cuda:2"))
                sent_num = int(G.number_of_nodes(ntype="sentence") / batch_size)
                labels = G.ndata['label']['sentence'].reshape(batch_size * sent_num * sent_num)
                model.train()
                initiate_scores = torch.tensor(batch_size * [[1 / sent_num] * sent_num]).cuda(2)
                # scores: [batch_size,sent_num]  s_h:[batch_size*sent_num, dim]  super_h:[batch_size,dim]
                # sent_pair:[batch_size*sent_num*sent_num, 2]
                forward_start_time = time.time()
                new_score, s_h, super_h, sent_pair = model.forward(G, initiate_scores, G_cls)
                logger.info('forward time: {:5.2f}s'.format(time.time() - forward_start_time))
                loss_start_time = time.time()
                loss = criterion(new_score, sent_pair, s_h, super_h, labels, sent_num)
                optimizer.zero_grad()
                loss.backward()
                logger.info('loss time: {:5.2f}s'.format(time.time() - loss_start_time))

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                train_loss += float(loss.data)
                epoch_loss += float(loss.data)

                if i % 100 == 0:
                    logger.info('       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.6f} | '
                                .format(i, (time.time() - iter_start_time), float(train_loss / 100)))
            except Exception as e:
                traceback.print_exc()

            continue

        epoch_avg_loss = epoch_loss / len(train_loader)
        logger.info('   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.6f} | '
                    .format(epoch, (time.time() - epoch_start_time), float(epoch_avg_loss)))

        if not best_train_loss or epoch_avg_loss < best_train_loss:
            save_file_path = os.path.join(train_dir, "bestmodel")
            logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s', float(epoch_avg_loss),
                        save_file_path)
            save_model(model, save_file_path)
            best_train_loss = epoch_avg_loss

        elif epoch_avg_loss >= best_train_loss:
            logger.error("[Error] training loss does not descent. Stopping supervisor...")
            save_model(model, os.path.join(train_dir, "earlystop"))
            sys.exit(1)

        # model, validate_loader, best_loss, best_F, non_descent_cnt, saveNo
        # 在每个train dataset epoch上训练完后就去验证集上验证loss值和标签F值和rouge值
        best_loss, best_F, non_descent_cnt, saveNo = validate(model, validate_loader, validate_dataset, best_loss,
                                                              best_F,
                                                              non_descent_cnt, saveNo, batch_size)

        if non_descent_cnt >= 3:
            logger.error("[Error] val loss does not descent for three times. Stopping supervisor...")
            save_model(model, os.path.join(train_dir, "earlystop"))
