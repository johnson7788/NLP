# -- encoding:utf-8 --


import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from nets.metric import Metrics
from nets import base_model


class Network(base_model.Network):

    def __init__(self, with_word2vec=False, vocab_size=None, embedding_dimensions=None,
                 embedding_table=None, train_embedding_table=False,
                 num_class=2, network_name="TextRNN", weight_decay=0.01,
                 optimizer_type="adam", optimizer_parameters_func=None, saver_parameters={'max_to_keep': 2},
                 attention_dimension_size=128, attention_layers=3, attention_headers=16, *args, **kwargs):
        """
        :param with_word2vec: 是否使用Word2Vec训练好的转换参数作为Embedding Lookup的参赛值
        :param vocab_size:  词汇数目
        :param embedding_dimensions: Embedding Loopup转换的时候，单词转换的词向量大小
        :param embedding_table: 训练好的单词向量映射表
        :param train_embedding_table: 是否训练train_embedding_table的参数值
        :param num_class:  类别数目
        :param network_name:  网络名称
        :param weight_decay: L2正则项的系数
        :param optimizer_type: 优化器的类别
        :param optimizer_parameters_func: 构建优化器的参数的函数
        :param saver_parameters: 模型持久化器的参数
        :param attention_dimension_size: Self Attention计算过程中的维度大小
        :param attention_layers: RNN的层次
        :param attention_headers: 头的数目
        """
        self.attention_dimension_size = attention_dimension_size
        self.attention_layers = attention_layers
        self.attention_headers = attention_headers

        super(Network, self).__init__(with_word2vec=with_word2vec, vocab_size=vocab_size,
                                      embedding_dimensions=embedding_dimensions,
                                      embedding_table=embedding_table, train_embedding_table=train_embedding_table,
                                      num_class=num_class, network_name=network_name, weight_decay=weight_decay,
                                      optimizer_type=optimizer_type,
                                      optimizer_parameters_func=optimizer_parameters_func,
                                      saver_parameters=saver_parameters)

    def interface(self):
        """
        前向网络构建
        batch_size: N
        feature height: H, 将序列长度T认为是H
        feature width: W，将Embedding size大小认为是W
        feature channel : C，一个文本就相当于一个Feature Map，通道数为1
        sentence_length: T
        embedding size: E
        :return:
        """
        with tf.variable_scope(self.network_name):
            with slim.arg_scope(self.arg_score()):
                with tf.variable_scope("placeholders"):
                    self.global_step = tf.train.get_or_create_global_step()
                    # 输入的单词id，形状为:[N,T]
                    self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_word_id')
                    # 希望输出的类别id, 形状为:[N,]
                    self.targets = tf.placeholder(dtype=tf.int32, shape=[None], name='target_class_id')
                    # Dropout
                    self.dropout_keep_prob = tf.placeholder_with_default(1.0, shape=[], name='dropout_keep_prob')

                # 1. Embedding Layer(N,T,E)
                embedding_inputs = self.embedding_lookup(self.inputs)

                # 2. 使用Transformer来提取高阶特征
                with tf.variable_scope("transformer"):
                    with tf.variable_scope("Input"):
                        encoder_input = tf.layers.dense(embedding_inputs, units=self.attention_dimension_size,
                                                        activation=tf.nn.relu)

                    for layer in range(self.attention_layers):
                        with tf.variable_scope("Encoder_{}".format(layer)):
                            # 1. 得到各个头的信息
                            attention_outputs = []
                            for header in range(self.attention_headers):
                                with tf.variable_scope("Header_{}".format(header)):
                                    attention_output = self._self_attention(
                                        H=encoder_input,
                                        attention_dimension_size=self.attention_dimension_size
                                    )
                                    attention_outputs.append(attention_output)

                            # 2. 拼接
                            attention_output = tf.concat(attention_outputs, axis=-1)

                            # 3. 做一个线性转换
                            attention_output = tf.layers.dense(attention_output,
                                                               units=self.attention_dimension_size,
                                                               activation=None)

                            # 4. 将当前层的输出和当前层的输入做一个残差结构
                            attention_output = tf.nn.relu(attention_output + encoder_input)

                            # 5. 将当前层输出作为下一层的输入
                            encoder_input = attention_output

                # 3. 将高阶特征拼接到一起,作为CNN提取出来的最终高阶特征信息
                with tf.variable_scope("merge_feature"):
                    # 4. 将所有时刻的特征信息求均值
                    features = tf.reduce_mean(attention_output, axis=1)

                # 4. FFN+Softmax做最终的决策输出
                with tf.variable_scope("project"):
                    score = slim.fully_connected(features, num_outputs=self.num_class, activation_fn=None)
                    # 重命名, 得到的是N个文本属于num_class个类别的置信度
                    self.logits = tf.identity(score, 'logits')
                    # 得到N个文本分别属于各个类别的概率值
                    self.probability = tf.nn.softmax(self.logits, name='probability')
                    # 得到最终的预测id
                    self.predictions = tf.argmax(self.logits, axis=-1, name='predictions')

        # 配置一个参数表示仅恢复模型参数
        self.saver_parameters['var_list'] = tf.global_variables()

    def _self_attention(self, H, attention_dimension_size):
        """
        计算Self-Attention
        :param H: [N,T,E]， N个序列，每个序列T个时刻，每个时刻E维的向量
        :return:
        """
        # 0. 获取大小信息
        hidden_size = H.shape[-1]
        batch_size, sequence_length, _ = tf.unstack(tf.shape(H))
        # 1. 对输入数据reshape操作
        H = tf.reshape(H, shape=tf.stack([batch_size * sequence_length, hidden_size]))
        # 2. 分别计算Q、K、V
        Q = tf.layers.dense(H, units=attention_dimension_size)
        K = tf.layers.dense(H, units=attention_dimension_size)
        V = tf.layers.dense(H, units=attention_dimension_size, activation=tf.nn.relu)
        # 3. Reshape
        Q = tf.reshape(Q, shape=tf.stack([batch_size, sequence_length, attention_dimension_size]))
        K = tf.reshape(K, shape=tf.stack([batch_size, sequence_length, attention_dimension_size]))
        V = tf.reshape(V, shape=tf.stack([batch_size, sequence_length, attention_dimension_size]))
        # 4. 计算相关性([N,T,E],[N,T,E],F,T) --> [N,T,T]
        scores = tf.matmul(Q, K, False, True) / np.sqrt(attention_dimension_size)
        # 5. 计算概率值([N,T,T])
        weights = tf.nn.softmax(scores)
        # 6. 计算最终结果
        attention = tf.matmul(weights, V)
        return attention












