# -- encoding:utf-8 --


import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from nets.metric import Metrics
from nets import base_model


class Network(base_model.Network):

    def __init__(self, with_word2vec=False, vocab_size=None, embedding_dimensions=None,
                 embedding_table=None, train_embedding_table=False,
                 num_class=2, network_name="TextCNNRNN", weight_decay=0.01,
                 optimizer_type="adam", optimizer_parameters_func=None, saver_parameters={'max_to_keep': 2},
                 num_filters=128, region_sizes=[2, 3, 4],
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
        :param num_filters: TextCNN 各个不同类型卷积核的数目，可以给定为int或者list
        :param region_sizes: TextCNN各个不同类别卷积核提取单词特征的单词数量范围
        :param attention_dimension_size: Self Attention计算过程中的维度大小
        :param attention_layers: Transformer的层次
        :param attention_headers: 头的数目
        """
        self.attention_dimension_size = attention_dimension_size
        self.attention_layers = attention_layers
        self.attention_headers = attention_headers
        self.region_sizes = region_sizes  # 使用CNN提取特征信息的时候，提取范围大小
        if isinstance(num_filters, list):
            # 相当于针对每个范围给定不同的卷积核数目
            if len(region_sizes) != len(num_filters):
                raise Exception("resize_sizes和num_filters大小必须一致!!!")
            else:
                self.num_filters = num_filters
        elif isinstance(num_filters, int):
            self.num_filters = [num_filters] * len(region_sizes)
        else:
            raise Exception("参数num_filters仅支持int类型或者list类型数据!!")

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

                # 1. Embedding Layer
                # 将单词id转换为单词向量，[N,T] --> [N,T,E]
                embedding_inputs = self.embedding_lookup(self.inputs)
                # 增加维度信息，将其转换为四维对象, [N,T,E] --> [N,T,E,1]
                expanded_embedding_inputs = tf.expand_dims(embedding_inputs, axis=-1)

                # 2. 使用卷积来提取高阶特征
                outputs = []
                with tf.variable_scope("cnn"):
                    for idx, region_size in enumerate(self.region_sizes):
                        with tf.variable_scope("conv-max-pooling-{}".format(idx)):
                            conv2d_input = expanded_embedding_inputs
                            # 卷积的功能相当于将region_size个单词看成一个整体，然后进行单词的特征向量信息的融合提取
                            # 最终返回结果形状为: [N,T,1,C]
                            # 为了保障卷积之后的Feature Map大小和原始大小一致(序列长度一致)，所以这里进行数据的填充
                            if region_size - 1 != 0:
                                top = (region_size - 1) // 2
                                bottom = region_size - 1 - top
                                conv2d_input = tf.pad(conv2d_input, paddings=[[0, 0], [top, bottom], [0, 0], [0, 0]])
                            # 卷积(序列长度不变)
                            conv = slim.conv2d(
                                conv2d_input,  # [N,T,E,1]
                                num_outputs=self.num_filters[idx],  # C, eg:2
                                kernel_size=(region_size, self.embedding_dimensions)  # (h,w), eg:(3,E)
                            )

                            # 最大池化(变成原来的一半), 将相邻单词的特征提取主要特征信息
                            pool = slim.max_pool2d(conv, (3, 1), stride=[2, 1])

                            # 添加到临时列表中
                            outputs.append(tf.squeeze(pool, axis=2))

                with tf.variable_scope("transformer"):
                    with tf.variable_scope("Input"):
                        encoder_input = tf.layers.dense(tf.concat(outputs, axis=-1),
                                                        units=self.attention_dimension_size,
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
