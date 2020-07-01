import time
import json
from typing import List
import re
import os
import jieba
from langconv import Converter
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora, similarities
from gensim.models.keyedvectors import Word2VecKeyedVectors
from transformers import AlbertModel, AlbertTokenizer, BertTokenizer,  AlbertForMaskedLM
import torch

######################################################
# 参数, 使用tfidf+doc2vec+albert实现无监督文本分类
#对内存影响较大的是文件读取, 可以改用迭代器(todo)，  影响运行时间的doc2vec训练次数，albert使用的模型
######################################################
# 加载nlpcc的中文数据
source_file = '/Users/admin/Downloads/nlpcc2017textsummarization/train_without_summ.txt'
#把nlpcc的数据过滤后放入des_file文件夹中
des_file = 'data/test/'
#停止词文件
stopword_file = 'stopwords.txt'
#经过预处理后的文件缓存位置
final_file = 'data/documents.txt'
#doc2vec 模型保存文章
docmodel = 'data/doc.model'
#tfidf模型保存位置
tfidfmodel = 'data/tfidf.model'
#人工定义的文章分类的类别，标签
finTags = ['明星', '诗歌', '故事', '美食', '企业', '个人', '证件', '新闻']
#停止词过滤
stopwords_list = [line.rstrip() for line in open(stopword_file, encoding="utf-8")]

def percent_chinese(sentence: str)-> bool:
    """
    过滤掉英文字符和数字占30%的文档
    :param sentence:
    :return:
    """
    #文本总的长度
    tol = len(list(sentence.split()))
    pattern = '[a-z0-9]+'
    #英文和数字的长度
    english_count = len(re.findall(pattern, sentence))
    return english_count/tol < 0.3

def filter_data():
    """
    过滤掉单词小于10000的文本，并且中文占比过低的文本, 保存到des_file文件夹
    :return:
    """
    count = 0
    with open(source_file) as f:
        for line in f:
            line_dict = json.loads(line)
            article = line_dict['article']
            if len(article) > 5000 and percent_chinese(article) :
                count += 1
                des = des_file + str(count) + '.txt'
                with open(des, 'w', encoding='utf-8') as wf:
                    wf.write(article + "\n")
    print('生成文档的个数',count)

def filter_chinese(sentence: str)-> str:
    """
    中文的一些预处理
    :param sentence: 输入的句子或文本
    :return:
    """
    # 去除文本中的url
    # sentence = re.sub(r"http\S+", "", sentence)
    #剔除所有数字
    # decimal_regex = re.compile(r"[^a-zA-Z]\d+")
    # sentence = decimal_regex.sub(r"", sentence)
    #删除英文字符
    # eng_regex = re.compile(r'[a-zA-z]')
    # sentence = eng_regex.sub(r"", sentence)
    #只保留中文和标点符号
    words = [word for word in sentence if word >= u'\u4e00' and word <= u'\u9fa5' or word in ['，','。','？','！']]
    sentence = ''.join(words)
    # 去除空格
    space_regex = re.compile(r"\s+")
    sentence = space_regex.sub(r"", sentence)
    # 繁体字转换成简体字
    sentence = Converter('zh-hans').convert(sentence)
    return sentence.strip().lower()

def jieba_segment(sentence: str)-> str:
    """
    jieba分词，并去掉停止词
    :param sentence:
    :return:
    """
    sentence_list = jieba.cut(sentence)
    sentence_list = [w for w in sentence_list if w not in stopwords_list]
    sentence = ' '.join(sentence_list)
    return sentence

def get_documents(cache=True, jieba=True)-> List:
    """
    返回所有文档预处理和jieba分词后的一个列表
    :param cache:  是否使用缓存的文件
    :param jieba:  是否进行分词
    :return:
    """
    documents = []
    #使用缓存文件
    if os.path.isfile(final_file) and cache:
        with open(final_file, 'r', encoding='utf-8') as file:
            for document in file:
                if jieba:
                    document = jieba_segment(document)
                documents.append(document)
    else:
        #读取要处理的文件列表
        desfiles = os.listdir(des_file)
        #处理后存入到final_file单个文件
        with open(final_file, 'w', encoding='utf-8') as wf:
            for des in desfiles:
                document = ''
                with open(des_file+des, 'r', encoding='utf-8', errors='ignore') as file:
                    for sentence in file:
                        sentence = filter_chinese(sentence)
                        if sentence:
                            document = document + sentence + '。'
                if document:
                    wf.write(document + "\n")
                    if jieba:
                        document = jieba_segment(document)
                    documents.append(document)
    print("文档的个数:",len(documents))
    return documents

def cal_tfidf(documents, topk=10)-> List:
    """
    tfidf模型训练
    :param documents: 要进行训练的文档
    :param topk: 提取tfidf score 的前多少个单词, 如果topk大于提取到的单词个数，返回所有单词
    :return:
    """
    # 单个文档分成列表
    docs = [[word for word in document.split(' ')] for document in documents]
    # 生成字典
    dictionary = corpora.Dictionary(docs)
    # 生成bag of word
    docs_bow = [dictionary.doc2bow(doc) for doc in docs]
    if os.path.isfile(tfidfmodel):
        model = TfidfModel.load(tfidfmodel)
    else:
        model = TfidfModel(docs_bow)
        model.save(tfidfmodel)
    # 生成文本向量
    docs_vector = list(model[docs_bow])
    # 对所有的文本向量进行排序，取钱topk
    docs_sort_vector = [sorted(doc, key=lambda x: x[1], reverse=True)[:topk] for doc in docs_vector]
    # 把对应的向量id转换成中文单词，docs_sort_chinese是中文单词和tfidf的score的列表
    docs_sort_chinese = [[(dictionary[vec[0]],vec[1]) for vec in doc] for doc in docs_sort_vector]
    return docs_sort_chinese

def albert_model(seq_length=510, model_name='voidful/albert_chinese_xxlarge'):
    """
    albert模型计算fintags和文档的相似度（使用余弦相似度)
    :param seq_length: 一个序列的最长长度
    :param model_name: 使用的albert的模型名称, 可选模型如下
                    voidful/albert_chinese_tiny
                    voidful/albert_chinese_small
                    voidful/albert_chinese_base
                    voidful/albert_chinese_large
                    voidful/albert_chinese_xlarge
                    voidful/albert_chinese_xxlarge
    :return: 返回所有文档和每个fintags的相似度列表
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = AlbertModel.from_pretrained(model_name)
    #不是用jieba分词
    docs = get_documents(cache=True, jieba=False)
    #用于保存所有tags的向量
    tags_cls = []
    for tag in finTags:
        #对单个单词encode，生成单词对应的字典的id，是逐个字的id
        tag_token = tokenizer.encode(tag, add_special_tokens=True)
        # 转变成tensor向量，并扩充一个batch_size维度
        tagid = torch.tensor(tag_token).unsqueeze(0)
        #获取模型的输出结果
        outputs = model(tagid)
        #获取hidden_states的向量
        last_hidden_states = outputs[0]
        #获取单词的cls向量，代表整个单词的向量
        tag_cls = last_hidden_states[:, :1, :].squeeze(1)
        tags_cls.append(tag_cls)
    # 初始化余弦相速度
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # 用于保存计算所有关键字和所有文档计算余弦相似度后的结果
    docs_similiarity = []
    for doc in docs:
        # 对每个文档进行拆分分长度固定的句子
        doc_tup = [doc[i:i + seq_length] for i in range(0, len(doc), seq_length)]
        # 对每个文档进行换成token id，如果最后一个句子不够512，就padding到512位
        doc_token = tokenizer.batch_encode_plus(doc_tup, pad_to_max_length=True)
        # 获取生成的token id
        docid = torch.tensor(doc_token['input_ids'])
        # 放入模型
        outputs = model(docid)
        # 获取隐藏层的状态
        last_hidden_states = outputs[0]
        # 获取文档的cls向量，维度是[batch_size, Hidden_dimension], 这个batch_size就是上面的一个文档拆出来的每个句子，Hidden_dimension是模型的输出维度
        doc_cls = last_hidden_states[:,:1,:].squeeze(1)
        #用于保存每个fintags对这个句子的余弦相似度，就是这个类别关键字和这个句子的相似度
        tags_similiarity = []
        for tag_cls in tags_cls:
            #计算余弦相似度,tag_cls的维度是[1,Hidden_dimension], doc_cls维度[batch_size, Hidden_dimension], tag_doc_simliarity的维度[batch_size]
            # tag_doc_simliarity 这个关键字和每个句子的余弦相似度
            tag_doc_simliarity = cos(tag_cls,doc_cls)
            # 对比这个关键字和所有句子，取最大相似度
            tags_similiarity.append(torch.mean(tag_doc_simliarity))
        docs_similiarity.append(tags_similiarity)
        #用于测试，否则太慢
        if len(docs_similiarity) ==10:
            break
    return docs_similiarity

def test_tfidf():
    """
    测试tfidf的效果
    :return: 输出结果
    """
    documents = get_documents()
    #取前20个tfidf分数最大的值
    res = cal_tfidf(documents, topk=100)
    #用于打印文档，有标点符号，比较好看
    documents = get_documents(cache=True, jieba=False)
    keywords = []
    for idx, doc in enumerate(res):
        #取出关键tfidf文档计算得到的的关键字
        docword = [vec[0] for vec in doc]
        # 如果我们自定义的关键字在tfidf关键字列表中，就打印出来
        tags = [tag for tag in finTags if tag in docword]
        if not tags:
            #如果没有和给定的类别关键字重合，打印tfidf给出的前3个关键字
            print('没有找到和给定关键字匹配的，取tfidf的前3个关键字')
            tags = docword[:3]
        print(f"tfidf计算的最接近的keyword是: {tags}, 文档是: {documents[idx]}")
        keywords.append(tags[0])
    print(keywords)
    return keywords

def train_doc2vec(documents, training=False, epoch=300):
    """
    训练doc2vec
    :param documents:预处理后的文档
    :param training:是否继续训练
    :param epoch: 训练次数
    :return:
    """
    # 单个文档分成列表
    docs = [[word for word in document.split(' ')] for document in documents]
    # 是否使用已缓存的模型
    if os.path.isfile(docmodel):
        model = Doc2Vec.load(docmodel)
    else:
        #使用TaggedDocument处理成文档和文档名称索引处理数据
        documents = [TaggedDocument(doc, tags = [i]) for i, doc in enumerate(docs)]
        model = Doc2Vec(documents, vector_size=100, window=6, min_count=1, workers=3, dm=1, negative=20, epochs=epoch)
        model.save(docmodel)
    #是否继续训练, 这里有bug，需要改进
    if training:
        documents = [TaggedDocument(doc, tags = [i]) for i, doc in enumerate(docs)]
        model.train(documents, total_examples=model.corpus_count, epochs=epoch)
    return model

def test_doc2vec():
    """
    测试doc2vec的效果
    :return: 输出结果
    """
    documents = get_documents(cache=True, jieba=True)
    #加载模型, training继续训练模型
    model = train_doc2vec(documents, training=True, epoch=200)
    #用于打印
    documents = get_documents(cache=True, jieba=False)
    # 过滤出给的关键字fintags不在字典中的词语 ，所以这个词语没有词向量，无法计算相似度
    filter_tags = [tag for tag in finTags if tag in model.wv]
    if finTags != filter_tags:
        print('给定的fintags这写关键字不在doc2vec生成的字典中, 请更改关键字或者扩充训练文档, 使得训练文档包含这个关键字', set(finTags) - set(filter_tags))
    tagsvec = model.wv[filter_tags]
    keywords = []
    for idx, doc in enumerate(documents):
        docvec = model.docvecs[idx]
        #计算所有tag与这个文档的相似度
        tagssim = Word2VecKeyedVectors.cosine_similarities(docvec, tagsvec)
        maxsim = max(tagssim)
        keyword = finTags[list(tagssim).index(maxsim)]
        print(f"doc2vec计算的最接近的keyword是: {keyword}, 相似度是: {maxsim}, 文档是: {doc}")
        keywords.append(keyword)
    print(keywords)
    return keywords

def test_albert():
    """
    测试albert模型的效果
    :return:
    """
    docs_similiarity = albert_model(model_name='voidful/albert_chinese_tiny')
    # docs_similiarity = albert_model(model_name='voidful/albert_chinese_base')
    #获取所有文档列表
    docs = get_documents(cache=True, jieba=False)
    keywords = []
    for idx, doc_similiarity in enumerate(docs_similiarity):
        #找出最高的相似度
        maxsim = max(doc_similiarity)
        #找出最高相似度所对应的单词
        keyword = finTags[doc_similiarity.index(maxsim)]
        print(f'albert计算后的结果最相似的标签是{keyword}, 相似度是:{maxsim}, 文档是: {docs[idx]}')
        keywords.append(keyword)
    print(keywords)
    return keywords

if __name__ == '__main__':
    # filter_data()
    # docs = get_documents(cache=False, jieba=True)
    # twords = test_tfidf()
    dwords = test_doc2vec()
    # awords = test_albert()
    # print(twords,dwords,awords)
