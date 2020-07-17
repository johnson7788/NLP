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
                 num_units=128, layers=3, *args, **kwargs):
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
        :param num_units: RNN Cell中的神经元数目
        :param layers: RNN的层次
        """
        self.num_units = num_units  # RNN Cell的神经元数目
        self.layers = layers  # RNN的层次
        self.embedding_inputs = None
        self.sequence_length = None

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
                    # 计算序列实际长度, 最终形状为:[N,]
                    self.sequence_length = tf.reduce_sum(tf.sign(tf.abs(self.inputs)), axis=-1)

                # 1. Embedding Layer
                self.embedding_inputs = self.embedding_lookup(self.inputs)

                # 2. 使用RNN来提取高阶特征
                with tf.variable_scope("rnn"):
                    # a. 定义RNN的cell构建函数
                    def cell(_units):
                        _cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=_units)
                        return tf.nn.rnn_cell.DropoutWrapper(cell=_cell, output_keep_prob=self.dropout_keep_prob)

                    # b. 构建前向的cell和反向cell
                    cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells=[cell(self.num_units) for _ in range(self.layers)])
                    cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells=[cell(self.num_units) for _ in range(self.layers)])

                    # c. 获取得到序列的输出向量
                    # 数据都是按照原始的从左往右的序列得到的最终特征
                    # (正向提取特征信息[N,T,E], 反向提取特征信息[N,T,E])，(正向最终的状态信息，反向最终的状态信息)
                    # 如果给定了序列的实际长度，那么在进行计算的时候，仅计算实际序列长度部分的内容，对于后面填充的内直接返回zero
                    (output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw,  # 前向的RNN Cell
                        cell_bw,  # 反向的RNN Cell
                        inputs=self.embedding_inputs,  # 输入值, [N,T,E]
                        dtype=tf.float32,  # 给定RNN状态初始化值的类型
                        sequence_length=self.sequence_length,  # 给定序列的实际长度(因为序列是经过填充的)
                    )

                # 3. 将高阶特征拼接到一起,作为CNN提取出来的最终高阶特征信息
                with tf.variable_scope("merge_feature"):
                    # 4. 直接将所有时刻的输出特征值mean作为最终特征信息(由于填充位置输出是zero，所以求均值不会产生影响)
                    # [N,T,E] --> [N,E] --> [N,E]
                    div_denominator = tf.reshape(tf.to_float(self.sequence_length), shape=(-1, 1))
                    features_fw = tf.div(tf.reduce_sum(output_fw, axis=1), div_denominator)
                    features_bw = tf.div(tf.reduce_sum(output_bw, axis=1), div_denominator)
                    features = tf.concat([features_fw, features_bw], axis=-1)
                    # TODO: 获取实际序列最后要给时刻的输出特征向量作为高阶向量(下周一做)

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

    def losses(self):
        with tf.name_scope("loss"):
            # 1. 调用父类获得正常的分类损失函数
            total_loss = super(Network, self).losses()
            # 2. 加入对抗学习部分的损失函数
            with tf.name_scope("perturLoss"):
                with tf.variable_scope(self.network_name, reuse=True):
                    # a. 在Embedding上加入噪声信息
                    pertur_embedding_inputs = self._add_perturbation(total_loss)
                    # b. 正常网络结构的构建
                    with tf.variable_scope("rnn"):
                        # a. 定义RNN的cell构建函数
                        def cell(_units):
                            _cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=_units)
                            return tf.nn.rnn_cell.DropoutWrapper(cell=_cell, output_keep_prob=self.dropout_keep_prob)

                        # b. 构建前向的cell和反向cell
                        cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells=[cell(self.num_units) for _ in range(self.layers)])
                        cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells=[cell(self.num_units) for _ in range(self.layers)])

                        # c. 获取得到序列的输出向量
                        # 数据都是按照原始的从左往右的序列得到的最终特征
                        # (正向提取特征信息[N,T,E], 反向提取特征信息[N,T,E])，(正向最终的状态信息，反向最终的状态信息)
                        # 如果给定了序列的实际长度，那么在进行计算的时候，仅计算实际序列长度部分的内容，对于后面填充的内直接返回zero
                        (output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                            cell_fw,  # 前向的RNN Cell
                            cell_bw,  # 反向的RNN Cell
                            inputs=pertur_embedding_inputs,  # 输入值, [N,T,E]
                            dtype=tf.float32,  # 给定RNN状态初始化值的类型
                            sequence_length=self.sequence_length,  # 给定序列的实际长度(因为序列是经过填充的)
                        )

                    # 3. 将高阶特征拼接到一起,作为CNN提取出来的最终高阶特征信息
                    with tf.variable_scope("merge_feature"):
                        # 4. 直接将所有时刻的输出特征值mean作为最终特征信息(由于填充位置输出是zero，所以求均值不会产生影响)
                        # [N,T,E] --> [N,E] --> [N,E]
                        div_denominator = tf.reshape(tf.to_float(self.sequence_length), shape=(-1, 1))
                        features_fw = tf.div(tf.reduce_sum(output_fw, axis=1), div_denominator)
                        features_bw = tf.div(tf.reduce_sum(output_bw, axis=1), div_denominator)
                        features = tf.concat([features_fw, features_bw], axis=-1)

                    # 4. FFN+Softmax做最终的决策输出
                    with tf.variable_scope("project"):
                        score = slim.fully_connected(features, num_outputs=self.num_class, activation_fn=None)

                    # 构建损失
                    if self.num_class == 2:
                        # 二分类，可以考虑使用sigmoid交叉熵损失函数
                        # 将id哑编码: [None,] --> [None,num_class]
                        labels = tf.one_hot(self.targets, depth=self.num_class)
                        # 计算损失:([None,num_class], [None,num_class]) --> [None,]
                        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=score)
                        # 所有样本损失合并求均值:[None,] --> []
                        perturLoss = tf.reduce_mean(loss)
                    else:
                        # 多分类，考虑使用softmax交叉熵损失函数
                        # 基于id和logits置信度直接计算损失: ([None,], [None,num_class]) --> [None,]
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=score)
                        # 所有样本损失合并求均值:[None,] --> []
                        perturLoss = tf.reduce_mean(loss)

                pass
            # 3. 合并损失
            total_loss = total_loss + perturLoss
            tf.summary.scalar('total_loss2', total_loss)
            tf.summary.scalar('pertur_loss', perturLoss)
            return total_loss

    def _add_perturbation(self, loss):
        """
        给词向量添加噪声信息
        :param loss:
        :return:
        """
        with tf.name_scope("add_noise"):
            # 求解loss关于embedding input值求解对应梯度值
            grad, = tf.gradients(loss, self.embedding_inputs,
                                 aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
            # 停止反向传播([N,T,E])
            grad = tf.stop_gradient(grad)
            # 计算噪声信息
            with tf.name_scope("noise"):
                # a. 求每个样本的梯度均值([N,T,E] --> [N,1,1])
                alpha = tf.reduce_mean(tf.abs(grad), axis=[1, 2], keep_dims=True) + 1e-12
                # b. 求解L2 norm值
                l2_norm = alpha * tf.sqrt(tf.reduce_mean(tf.pow(grad / alpha, 2), [1, 2], keep_dims=True) + 1e-6)
                # c. 将grad除以l2_norm
                x_unit = grad / l2_norm
                # d. 扩展一下数据
                perturb = x_unit * 2.0
            return self.embedding_inputs + perturb
