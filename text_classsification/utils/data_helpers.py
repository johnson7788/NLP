# -- encoding:utf-8 --

import numpy as np
import re


# 清洗字符串，字符切分
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9(),.!?，。？！、“”\'\`]", " ", string)  # 考虑到中文
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    基于给定的正例和负例文件路径加载数据
    :param positive_data_file:
    :param negative_data_file:
    :return:
    """
    # 1. 加载所有数据组成list列表
    positive = open(positive_data_file, 'rb').read().decode('utf-8')
    negative = open(negative_data_file, 'rb').read().decode('utf-8')

    # 2.数据的划分(转换成一个一个样本)
    positive = positive.split("\n")
    negative = negative.split("\n")

    # 3. 数据简单处理
    positive = [clean_str(s.strip()) for s in positive]
    negative = [clean_str(s.strip()) for s in negative]
    positive = [s for s in positive if len(s) > 0]
    negative = [s for s in negative if len(s) > 0]

    # 4. 数据合并得到x
    texts = positive + negative

    # 5. 得到对应的id
    labels = [1] * len(positive) + [0] * len(negative)

    # 6. 结果返回
    return np.asarray(texts), np.asarray(labels)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    基于给定的data数据获取批次数据
    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    # 一个epoch里面有多少个bachsize
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            # 传给permutation一个矩阵，它会返回一个洗牌后的矩阵副本
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    texts, labels = load_data_and_labels()
    # from utils.vocabulary_utils import VocabularyProcessorUtil, split_with_word
    #
    # _, vocabulary = VocabularyProcessorUtil.load_word2vec_embedding("../model/w2v.bin")
    # VocabularyProcessorUtil.building_model(documents=texts, save_path='../model/vocab.pkl', max_document_length=512,
    #                                        vocabulary=vocabulary,
    #                                        split_fn=split_with_word)
    # model = VocabularyProcessorUtil.load_model('../model/vocab.pkl')

