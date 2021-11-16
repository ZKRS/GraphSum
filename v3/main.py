import os
import sys
import time
import traceback

import gensim
import torch
import torch.nn.functional as F
from dgl.data.utils import load_graphs
from rouge import Rouge
from torch.utils.data import DataLoader

from modules.loss import FocalLoss, Loss
from modules.test import Tester
from modules.dataloader import DatasetGraph, PadBatchGraph
from modules.model import GraphConvolutionNet
from util import logger, f_logger, save_model, read_txt_data, get_vocab_embedding


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
        return

    # 计算抽取的句子与gold summary的rouge值
    rouge = Rouge()
    scores_all = rouge.get_scores(tester.candidate, tester.gold, avg=True)
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
    criterion = Loss(FocalLoss(hps), hps).cuda(hps.gpu)
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
                labels = G.ndata['label']['sentence'].reshape(hps.batch_size * sent_num * sent_num, -1)

                model.train()
                initiate_scores = torch.tensor(hps.batch_size * [[1 / sent_num] * sent_num]).cuda(hps.gpu)
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

                if hps.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), hps.max_grad_norm)
                optimizer.step()

                train_loss += float(loss.data)
                epoch_loss += float(loss.data)

                if i % 1000 == 0:
                    f_logger.info('       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.6f} | '
                                  .format(i, (time.time() - iter_start_time), float(train_loss / 1000)))
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
            sys.exit(1)

        # model, validate_loader, best_loss, best_F, non_descent_cnt, saveNo
        # 在每个train dataset epoch上训练完后就去验证集上验证loss值和标签F值和rouge值
        best_loss, best_F, non_descent_cnt, saveNo = validate(hps, model, validate_loader, validate_dataset, best_loss,
                                                              best_F, non_descent_cnt, saveNo)

        if non_descent_cnt >= 3:
            f_logger.error("[Error] val loss does not descent for three times. Stopping supervisor...")
            save_model(model, os.path.join(hps.train_dir, "earlystop"))

    # model.load_state_dict(torch.load('./model/train/bestmodel'))
    # best_loss, best_F, non_descent_cnt, saveNo = validate(model, validate_loader, validate_dataset, best_loss,
    #                                                       best_F,
    #                                                       non_descent_cnt, saveNo, batch_size)


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
            self.gpu = len(self.device_ids)

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
            self.loss_prior = 0.01
            self.loss_lambda = 0.5


            self.restore_model = "None"
            self.save_root = "./model"


    hps = Hps()

    # 加载词典，和词典嵌入，和过滤词
    vocab_freq = read_txt_data(hps.vocab_path)
    vocab = [v_f.split()[0] for v_f in vocab_freq]
    word_embedding = gensim.models.Word2Vec.load(hps.word_embedding_path)
    filter_word = read_txt_data(hps.filter_word_path)

    # 加载训练数据集和验证集
    dataset_start_time = time.time()
    train_dataset = DatasetGraph(load_graphs(hps.train_dataset_graphs_path)[0], hps.train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=hps.batch_size, shuffle=hps.shuffle,
                              num_workers=hps.num_workers,
                              collate_fn=PadBatchGraph(vocab), drop_last=True)
    logger.info('train dataset start time: {:5.2f}s'.format(time.time() - dataset_start_time))
    del train_dataset
    validate_dataset = DatasetGraph(load_graphs(hps.val_dataset_graphs_path)[0], hps.validate_data_path)
    validate_loader = DataLoader(validate_dataset, batch_size=hps.batch_size, shuffle=hps.shuffle,
                                 num_workers=hps.num_workers,
                                 collate_fn=PadBatchGraph(vocab), drop_last=True)

    # 词嵌入矩阵
    vocab_embedding = get_vocab_embedding(word_embedding, vocab, hps.w_input_dim)
    embeded = torch.nn.Embedding(len(vocab_embedding), hps.w_input_dim, padding_idx=0)
    embeded.weight.data.copy_(torch.Tensor(list(vocab_embedding.values())))
    embeded.weight.requires_grad = True
    embeded = embeded.cuda(hps.gpu)

    # 模型
    model = GraphConvolutionNet(hps.w_input_dim, hps.w_hidden_dim, hps.w_output_dim, hps.s_input_dim, hps.s_hidden_dim,
                                hps.s_output_dim,
                                embeded).cuda(hps.gpu)

    # train
    setup_training(hps, model, train_loader, validate_loader)
