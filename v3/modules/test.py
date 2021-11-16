import torch

from util import logger
from modules.loss import FocalLoss, Loss

class Tester:
    def __init__(self, hps):
        super(Tester, self).__init__()
        self.hps = hps
        self.criterion = Loss(FocalLoss(hps), hps.batch_size).cuda(hps.gpu)
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
        sent_num = int(G.number_of_nodes(ntype="sentence") / self.hps.batch_size)
        labels = G.ndata['label']['sentence'].reshape(self.hps.batch_size * sent_num * sent_num, -1).cuda(self.hps.gpu)
        initiate_scores = torch.tensor(self.hps.batch_size * [[1 / sent_num] * sent_num]).cuda(self.hps.gpu)
        # scores: [batch_size,sent_num]  s_h:[batch_size*sent_num, dim]  super_h:[batch_size,dim]
        # sent_pair:[batch_size*sent_num*sent_num, 2]
        score, s_h, super_h, sent_pair = model.forward(G, initiate_scores, G_cls)
        loss = self.criterion(score, sent_pair, s_h, super_h, labels, sent_num)
        self.loss += loss
        # G_list = dgl.unbatch(G)
        # G_cls_list = dgl.unbatch(G_cls)

        sent_pair = sent_pair.reshape(self.hps.batch_size, sent_num, sent_num, 2).max(3)[1].cpu()

        # 得到映射，填充candidate， gold的句子
        for i in range(self.hps.batch_size):  # 遍历batch中的每个item
            index = indexes[i]
            doc = validate_dataset.get_doc_item(index)
            labels = doc["label"]
            text = doc["text"]
            summary = doc["summary"]
            # self.gold.append([" ".join([text[label] for label in labels])])
            self.gold.append(" ".join(summary))
            candidate_labels = self.decode(sent_pair[i, :, :])
            logger.info("candidate_labels")
            logger.info(len(candidate_labels) == 0)
            self.candidate.append(" ".join([text[c_l] for c_l in candidate_labels]))
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

