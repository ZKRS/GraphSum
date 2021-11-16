import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.alpha = hps.loss_alpha
        self.gamma = hps.loss_beta

    def forward(self, predict, labels):
        """"
        二分类的focal loss
        :param predict: [batch_size*sent_num*sent_num, 2]
        :param labels: [batch_size*sent_num*sent_num]
        """
        pt = torch.gather(predict, dim=1, index=labels)
        print("label", labels.shape)
        pos = torch.full(labels.shape, 0.1, device=torch.device("cuda:" + str(self.hps.gpu)))
        neg = torch.full(labels.shape, 0.9, device=torch.device("cuda:" + str(self.hps.gpu)))
        self.alpha = torch.where(labels == 0, neg, pos)
        return torch.mean(-self.alpha * torch.pow((1 - pt), self.gamma) * pt.log())


class Loss(torch.nn.Module):
    def __init__(self, focal_loss, batch_size):
        super().__init__()
        self.focal_loss = focal_loss
        self.batch_size = batch_size
        self.T = 100

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

        focal_loss = self.focal_loss(sent_pair, labels.long())
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
        return focal_loss + torch.exp(-1 * sim_s_super) + sim_s_s
