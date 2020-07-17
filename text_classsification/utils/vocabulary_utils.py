# -- encoding:utf-8 --

import os
import itertools
import jieba
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import preprocessing
from gensim import utils
from gensim.models import word2vec


def default_split_fn(documents):
    return split_with_char(documents)


def split_with_char(documents):
    return [list(sentence) for sentence in documents]


def split_with_word(documents):
    return [list(filter(lambda word: len(word) > 0, jieba.cut(sentence.strip()))) for sentence in documents]


class CategoricalVocabulary(preprocessing.CategoricalVocabulary):
    def __init__(self, unknown_token="<UNK>"):
        super(CategoricalVocabulary, self).__init__(unknown_token, False)

        # 特殊值（填充0，未知1）
        self.padding_token = "<PAD>"
        self._mapping[self.padding_token] = 0
        self._mapping[self._unknown_token] = 1
        # 添加一个属性
        self.vocab_size = 2

    def get(self, category):
        if category not in self._mapping:
            return 1
        return self._mapping[category]

    def set(self, category, index):
        self._mapping[category] = index
        self.vocab_size += 1


class PathLineSentences(object):
    """Like :class:`~gensim.models.word2vec.LineSentence`, but process all files in a directory
    in alphabetical order by filename.

    The directory must only contain files that can be read by :class:`gensim.models.word2vec.LineSentence`:
    .bz2, .gz, and text files. Any file not ending with .bz2 or .gz is assumed to be a text file.

    The format of files (either text, or compressed text files) in the path is one sentence = one line,
    with words already preprocessed and separated by whitespace.

    Warnings
    --------
    Does **not recurse** into subdirectories.

    """

    def __init__(self, source, max_sentence_length=word2vec.MAX_WORDS_IN_BATCH, limit=None, split_fn=None):
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit
        if split_fn is None:
            self.split_fn = default_split_fn
        else:
            self.split_fn = split_fn

        if os.path.isfile(self.source):
            self.input_files = [self.source]  # force code compatibility with list of files
        elif os.path.isdir(self.source):
            self.source = os.path.join(self.source, '')  # ensures os-specific slash at end of path
            self.input_files = os.listdir(self.source)
            self.input_files = [self.source + filename for filename in self.input_files]  # make full paths
            self.input_files.sort()
        else:
            raise ValueError('input is neither a file nor a path')

    def __iter__(self):
        """iterate through the files"""
        for file_name in self.input_files:
            with utils.open(file_name, 'rb') as fin:
                for line in itertools.islice(fin, self.limit):
                    line = self.split_fn([utils.to_unicode(line).strip()])[0]
                    i = 0
                    while i < len(line):
                        yield line[i:i + self.max_sentence_length]
                        i += self.max_sentence_length


class VocabularyProcessorUtil(object):

    @staticmethod
    def building_model(documents, save_path, max_document_length=512, vocabulary=None, split_fn=default_split_fn):
        """
        基于传入的文档数据构建字典相关信息
        :param documents: 进行模型训练的时候的文本数据
        :param save_path: 模型持久化的路径
        :param vocabulary: 词汇映射表
        :param split_fn: 将文本转换为单词过程中的函数, 默认是将每个字当作一个单词
        :param max_document_length: 将文本单词id转换的时候，最长文本允许的单词数目
        :return:
        """
        tf.logging.info("开始构建词汇转换模型.....")
        model = preprocessing.VocabularyProcessor(max_document_length=max_document_length,
                                                  vocabulary=vocabulary, tokenizer_fn=split_fn)
        model.fit(raw_documents=documents)
        tf.logging.info("词汇转换模型构建完成，开始模型保存操作!!!")
        model.save(save_path)
        tf.logging.info("词汇转换模型保存完成，保存位置为:{}".format(save_path))

    @staticmethod
    def load_model(save_path) -> preprocessing.VocabularyProcessor:
        """
        基于给定的路径加载模型并返回
        :param save_path:
        :return:
        """
        if os.path.exists(save_path):
            tf.logging.info("从【{}】位置进行词汇转换模型的恢复!!!".format(save_path))
            return preprocessing.VocabularyProcessor.restore(save_path)
        else:
            raise Exception("词汇转换模型不存在，请检查磁盘路径：{}".format(save_path))

    @staticmethod
    def build_word2vec_embedding(data_path, save_path, embedding_dimensions):
        """
        基于data_path下的文件内容构建Word2Vec向量，并将向量保存到save_path这个路径中
        :param data_path: 原始数据所在的文件夹路径
        :param save_path: 训练好的数据保存路径
        :param embedding_dimensions:  转换的Embedding向量大小
        :return:
        """
        # 0. 加载数据
        sentences = PathLineSentences(source=data_path, split_fn=split_with_word)
        # 1. 构建Word2Vec模型
        model = word2vec.Word2Vec(sentences=sentences, size=embedding_dimensions,
                                  window=9, min_count=2, iter=50)
        # 3. 模型保存(以文本形式保存)
        model.wv.save_word2vec_format(fname=save_path, binary=True)

    @staticmethod
    def load_word2vec_embedding(save_path):
        """
        加载Word2Vec训练好的embedding转换矩阵
        :param save_path:  数据存储的路径
        :param binary: 是否是二进制存储
        :return: embedding_table, vocabulary
        """
        # 1. 加载数据
        model = word2vec.Word2VecKeyedVectors.load_word2vec_format(save_path, binary=True)
        # 2. 获取embedding_table
        embedding_table = model.vectors
        embedding_dimensions = np.shape(embedding_table)[1]
        # 3. 获取单词和id之间的映射关系
        vocabulary = CategoricalVocabulary()
        vocab_size = vocabulary.vocab_size
        for word in model.vocab:
            vocabulary.set(word, model.vocab[word].index + vocab_size)
        # 4. 在embedding_table前面加入特征字符所代表的含义
        embedding_table = np.concatenate(
            [
                np.zeros(shape=(1, embedding_dimensions), dtype=embedding_table.dtype),  # PAD对应的的特征值
                np.random.normal(0, 0.01, size=(1, embedding_dimensions)),  # UNK对应的特征值
                embedding_table  # 原始单词对应的特征值
            ],
            axis=0
        )
        return embedding_table, vocabulary


if __name__ == '__main__':
    VocabularyProcessorUtil.build_word2vec_embedding("../data", "../model/w2v2.bin", 128)
    embedding_table, vob = VocabularyProcessorUtil.load_word2vec_embedding("../model/w2v.bin")
    print(vob.vocab_size)
