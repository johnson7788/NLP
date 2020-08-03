import pickle
import os
import torchtext
import jieba
from typing import List

class TextDataset(torchtext.data.Dataset):
    """
    读取数据并处理
    """
    @staticmethod
    def sort_key(example):
        """
        用于在torchtext.data.Iterator生成批次迭代器的时候，用于example进行排序以将具有相似长度的example批次放在一起并最小化填充
        如果在使用torchtext.data.Iterator时，提供了sort_key，那么就会覆盖这个Dataset的sort_key属性, 默认为None
        :param example:  是按example中单个样本的text属性的长度排序
        :return:
        """
        return len(example.text)
    def __init__(self, path, text_field, label_field, **kwargs):
        """根据给的Field和数据集路径处理数据, 之后交给torchtext.data.Dataset处理
        Arguments:
            path: 数据集的路径
            text_field: text数据的Field格式
            label_field: label数据的Field格式
            **kwargs: data.Dataset的参数
        """
        #定义fields
        fields = [('text', text_field), ('label', label_field)]
        #定义一个空的数据集
        examples = []
        #列出当前目录下的所有文件夹，文件夹名称作为label，文件夹里面的文件内容作为text
        dirname = os.listdir(path)
        for dir in dirname:
            #循环一个label目录下的所有文件
            files = os.listdir(path + '/' + dir)
            for file in files:
                document = ''
                with open(path + '/' + dir + '/' + file, encoding="utf8", errors='ignore') as f:
                    for line in f:
                        if line != '\n':
                            document += text_filter(line)
                # 如果文本长度小于10个字符，那么就过滤掉
                if len(document) < 10:
                    continue
                text, label = document, dir
                #定义一个训练或测试的样本的Example格式
                example = torchtext.data.Example()
                # text_field.preprocess 是进行token的处理, 例如用jieba处理
                setattr(example, "text", text_field.preprocess(text))
                setattr(example, "label", label_field.preprocess(label))
                examples.append(example)
        super(TextDataset, self).__init__(examples, fields, **kwargs)

def text_filter(sentence:str)-> str:
    """
    过滤掉非汉字和标点符号和非数字
    :param sentence:
    :return:
    """
    line = sentence.replace('\n', '。')
    # 过滤掉非汉字和标点符号和非数字
    linelist = [word for word in line if
                word >= u'\u4e00' and word <= u'\u9fa5' or word in ['，', '。', '？', '！',
                                                                    '：'] or word.isdigit()]
    return ''.join(linelist)

#定义text的Field
def text_token(sentence: str)-> List:
    """
    使用jieba分词
    :param sentence: 要分词的sentence
    :return: 一个text的分词后的列表
    """
    return jieba.lcut(sentence)

#sequential 是否要变成序列，tokenize表示使用的token 函数是， lower表示是否转换成小写
TextTEXT = torchtext.data.Field(sequential=True, tokenize=text_token, lower=True)

#定义label的Field
TextLABEL = torchtext.data.Field(sequential=False, lower=True)

def text_dataloader(path, batch_size, shuffle=False):
    """
    加载数据
    :param path:  训练集和测试集的文件路径
    :param batchsize: 批处理大小
    :param shuffle: 是否做shuffle
    :return:
    """
    #定义text和label的 Field格式
    text_field = TextTEXT
    label_field = TextLABEL

    #读取数据
    #dataset 包含examples和fields2部分,examples保存所有的数据，field是这类数据的名字，例如field是(text,label), examples里面就是[(label的内容(纯文本),text内容(纯文本)),...]
    dataset = TextDataset(path, text_field, label_field)
    #构建字典,使用build_vocab之后text_field会多出一个vocab的属性，vocab中是字典
    text_field.build_vocab(dataset)
    label_field.build_vocab(dataset)
    #创建迭代器
    dataiter = torchtext.data.Iterator(dataset, batch_size, shuffle=shuffle, repeat=False)
    return dataiter, text_field, label_field

def save_vocab(vocab, filename):
    """
    使用pickle保存字典
    :param vocab:
    :param filename:
    :return:
    """
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(filename):
    """
    使用pickle加载字典
    :param filename:
    :return:
    """
    with open(filename, 'rb') as f:
        vocab = pickle.load(f)
    return vocab
