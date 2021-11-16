import torch
import logging
from modules.textrank import TextRank
import numpy as np
import spacy
import json
import sys

formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

log_file_path = "./log/train.log"
f_logger = logging.getLogger("graph sum logger")
file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf8')
file_handler.formatter = formatter
file_handler.setLevel(logging.INFO)
f_logger.addHandler(file_handler)
f_logger.setLevel(logging.DEBUG)

logger = logging.getLogger("Summarization logger")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

spacy_nlp = spacy.load('en_core_web_sm')


def save_model(model, save_file_path):
    with open(save_file_path, 'wb') as f:
        torch.save(model.state_dict(), f)
    logger.info('[INFO] Saving model to %s', save_file_path)


def read_json_data(data_path):
    logging.info('reading json data...' + data_path)
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def read_txt_data(data_path):
    logging.info('reading txt data...' + data_path)
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(line)
    return data


def preprocess(text, filter_words):
    """
    handle single document of multi-sentences, remove stop words and then extract entities
    :param text:
    :param filter_words:
    :return: list of sentences without stop words and entities from the text
    """
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    # 是否添加过滤词
    for w in filter_words:
        spacy_nlp.vocab[w].is_stop = True

    processed_text = []

    # remove stop words
    for sent in text:
        sent = spacy_nlp(sent)
        sent_tokens = [token.text for token in sent if not token.is_stop]
        processed_text.append(" ".join(sent_tokens))
    # get entities
    processed_text_str = " ".join(processed_text)
    processed_text_str = spacy_nlp(processed_text_str)
    entities = [ent.text.strip() for ent in processed_text_str.ents]

    return processed_text, set(entities)


# 单文档用textrank， TODO: 多文档用lda
def extract_keyword(text, steps=20, top_k=20, allow_pos=['NOUN', 'PROUN']):
    """
    extract keyword from doc
    :param text:
    :param steps:
    :param top_k:
    :param allow_pos:
    :return: top k keywords
    """
    logging.info('extract keyword...')
    tr = TextRank(steps)
    tr.analyze(text, candidate_pos=allow_pos, window_size=4, lower=False)
    keywords_score = tr.get_keywords(
        top_k)  # [('request', 1.281397222222222), ('pictures', 1.281397222222222), ('youtube', 1.0959555555555553), ('prosecutor', 1.0959555555555553), ('twitter', 0.9075388888888888), ('siege', 0.9075388888888888), ('access', 0.7151083333333333), ('week', 0.7151083333333333)]
    keywords = set()
    for ks in keywords_score:
        keywords.add(ks[0])

    return keywords


# 统计文本词频, 经过预处理后的：去除停用词和过滤词，TODO: 需要进行词干词形还原吗？
def cal_word_freq(text, words):
    logging.info('calculate word freq.')
    word_freq = {}
    # if re.match(r'[a-zA-Z0-9]*', text):  # 避免中文影响
    #     word_box.extend(text.strip().split())
    # word_freq = collections.Counter(word_box)
    for word in words:
        word_freq[word] = text.count(word)
    return word_freq


PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'


def get_word_id(vocab):
    word_id = {}
    for wid, w in enumerate([PAD_TOKEN, UNKNOWN_TOKEN] + vocab):
        word_id[w] = wid

    return word_id


def get_sent_id(text, word_id):
    sents_id = []
    for sent in text:
        sent_id = []
        for w in sent.split(" "):
            sent_id.append(word_id.get(w, word_id[UNKNOWN_TOKEN]))
        sents_id.append(sent_id)

    return sents_id


def padding_sent_matrix_id(sent_matrix_id, PAD_TOKEN_ID):
    sent_max_len = max([len(sent) for sent in sent_matrix_id])
    pad = [sent + (sent_max_len - len(sent)) * [PAD_TOKEN_ID] for sent in
           sent_matrix_id]  # TODO: 可能有问题 *操作, PAD_TOKEN_ID是int
    return pad


def get_vocab_embedding(word_embedding, vocab, w_input_dim):
    vocab_embedding = {}
    vocab = [PAD_TOKEN, UNKNOWN_TOKEN] + vocab
    for index, word in enumerate(vocab):
        if word == PAD_TOKEN:
            vocab_embedding[index] = [0] * w_input_dim
        elif word == UNKNOWN_TOKEN:
            vocab_embedding[index] = np.random.uniform(-1, 1, w_input_dim)
        else:
            if word in word_embedding.wv:
                vocab_embedding[index] = word_embedding.wv[word]

    return vocab_embedding
