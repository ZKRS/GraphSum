import json
import logging
import os
from gensim.models import word2vec


#
# def read_json_data(data_path):
#     data = []
#     with open(data_path, "r", encoding="utf-8") as f:
#         for line in f:
#             data.append(json.loads(line)['text'])
#
#     return data
#
#
# data = read_json_data('./data/cnndm/train.label.jsonl')
# data.extend(read_json_data('./data/cnndm/val.label.jsonl'))
def merge(file1, file2):
    f1 = open(file1, 'a+', encoding='utf-8')
    with open(file2, 'r', encoding='utf-8') as f2:
        f1.write('\n')
        for i in f2:
            f1.write(i)


def word_embedding(data_path, model_path):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence(data_path)
    model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=300)
    model.save(model_path)

def sent_embedding():
    pass


if __name__ == '__main__':
    data_path = './data/cnndm/train.txt'
    model_path = './model/cnndm_word2vec.model'

    # merge('./data/cnndm/train.txt', './data/cnndm/val.txt')
    word_embedding(data_path, model_path)
