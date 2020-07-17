# -- encoding:utf-8 --

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from nets.metric import Metrics


class Network(object):

    def __init__(self, with_word2vec=False, vocab_size=None, embedding_dimensions=None,
                 embedding_table=None, train_embedding_table=False,
                 num_class=2, network_name="TextCNN", weight_decay=0.01,
                 optimizer_type="adam", optimizer_parameters_func=None, saver_parameters={'max_to_keep': 2},
                 *args, **kwargs):
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
        """
        self.with_word2vec = with_word2vec
        self.weight_decay = weight_decay  # 正则的权重系数
        self.network_name = network_name  # 网络名称
        self.vocab_size = vocab_size  # 词汇表大小
        self.embedding_dimensions = embedding_dimensions  # 做单词id转换为向量的时候，向量维度大小
        self.input_embedding_table = embedding_table
        self.train_embedding_table = train_embedding_table
        self.num_class = num_class  # 类别数目

        if self.with_word2vec:
            if self.input_embedding_table is None or np.ndim(self.input_embedding_table) != 2:
                tf.logging.warn("当参数with_word2vec为True的时候，必须给定embedding_table的2维转换矩阵值!!")
                self.with_word2vec = False
            else:
                self.vocab_size, self.embedding_dimensions = np.shape(self.input_embedding_table)
        else:
            if self.embedding_dimensions is None or self.vocab_size is None:
                raise Exception("当参数with_word2vec为False的时候，必须给定embedding_dimensions和vocab_size的参数值!!")

        self.global_step = None  # Tensor变量对象，用于记录模型的更新次数
        self.embedding_table = None  # 做词嵌入的变量
        self.inputs = None  # 输入的文本单词id，[None,None]
        self.targets = None  # 实际标签下标对象, [None,]
        self.dropout_keep_prob = None  # Drouout系数
        self.logits = None  # 模型前向网络执行之后得到的预测置信度信息，[None, num_class]
        self.probability = None  # 模型前向网络执行之后得到的预测概率信息, [None, num_class]
        self.predictions = None  # 模型前向网络的预测结果/类别下标，[None,]
        self.saver = None  # 模型持久化的对象
        self.saver_parameters = saver_parameters  # 初始化模型持久化对象的参数

        self.optimizer_type = optimizer_type  # 优化器类型
        self.optimizer_parameters_func = optimizer_parameters_func  # 优化器参数

        self.interface()

    def arg_score(self):
        """
        作用域默认参数给定
        :return:
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='VALID', stride=1) as sc:
                return sc

    def embedding_lookup(self, inputs):
        """
        对输入做一个Embedding转换处理
        :param inputs: 输入的Tensor对象
        :return:
        """
        with tf.device('/cpu:0'), tf.variable_scope('embedding'):
            if self.with_word2vec:
                tf.logging.info("Embedding Table初始化使用Word2Vec训练好的转换参数.....")
                _embedding = tf.get_variable(
                    name='embedding_table',
                    shape=[self.vocab_size, self.embedding_dimensions],
                    initializer=tf.constant_initializer(value=self.input_embedding_table),
                    trainable=self.train_embedding_table  # 给定是否参与模型训练
                )
            else:
                tf.logging.info("Embedding Table初始化使用随机初始化值.....")
                _embedding = tf.get_variable(name='embedding_table',
                                             shape=[self.vocab_size, self.embedding_dimensions])
            self.embedding_table = _embedding
            # 将单词id转换为单词向量，[N,T] --> [N,T,E]
            embedding_inputs = tf.nn.embedding_lookup(self.embedding_table, inputs)
        return embedding_inputs

    def interface(self):
        raise NotImplementedError("请实现具体的interface代码，用于构建前向网络结构!!!")

    def losses(self):
        """
        计算损失函数，并返回对应的Tensor对象值
        基于预测的置信度logits以及实际的标签值来构建分类损失函数
        :return:
        """
        with tf.name_scope("Loss"):
            # 1. 计算实际值和预测值之间差值所导致的损失值
            if self.num_class == 2:
                # 二分类，可以考虑使用sigmoid交叉熵损失函数
                # 将id哑编码: [None,] --> [None,num_class]
                labels = tf.one_hot(self.targets, depth=self.num_class)
                # 计算损失:([None,num_class], [None,num_class]) --> [None,]
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=self.logits)
                # 所有样本损失合并求均值:[None,] --> []
                loss = tf.reduce_mean(loss)
            else:
                # 多分类，考虑使用softmax交叉熵损失函数
                # 基于id和logits置信度直接计算损失: ([None,], [None,num_class]) --> [None,]
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits)
                # 所有样本损失合并求均值:[None,] --> []
                loss = tf.reduce_mean(loss)

            # 2. 将损失添加到collection中
            tf.losses.add_loss(loss)

            # 3. 获取所有损失合并之后的值(分类损失、正则损失等等)
            total_loss = tf.losses.get_total_loss(name='total_loss')

            # 4. 可视化操作
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('total_loss', total_loss)
        return total_loss

    def optimizer(self, loss=None, return_train_operation=True):
        """
        构建优化器，并根据参数return_train_operation决定是否返回训练对象
        :param loss: 如果return_train_operation为True，那么loss参数必须有值，并且表示为损失值
        :param return_train_operation: True or False，True表示返回训练对象，False表示不返回
        :return:  如果return_train_operation为True，返回优化器以及训练操作对象，否则仅返回优化器本身
        """
        if return_train_operation and loss is None:
            raise Exception("当需要返回训练对象的时候，loss参数必须有值!!")

        with tf.name_scope("optimizer"):
            # 1. 构建优化器
            parameters = self.optimizer_parameters_func(self.global_step)
            if self.optimizer_type == 'adam':
                opt = tf.train.AdamOptimizer(**parameters)
            elif self.optimizer_type == 'adadelta':
                opt = tf.train.AdadeltaOptimizer(**parameters)
            elif self.optimizer_type == 'adagrad':
                opt = tf.train.AdagradOptimizer(**parameters)
            elif self.optimizer_type == 'ftrl':
                opt = tf.train.FtrlOptimizer(**parameters)
            elif self.optimizer_type == 'momentum':
                opt = tf.train.MomentumOptimizer(**parameters)
            else:
                opt = tf.train.GradientDescentOptimizer(**parameters)

            # 2. 构建训练对象
            train_op = None
            if return_train_operation:
                train_op = opt.minimize(loss=loss, global_step=self.global_step)
        return opt, train_op

    def metrics(self, loss=None):
        """
        构建模型的评估指标，并返回对象
        :param loss:
        :return:
        """

        def accuracy(true_y, pre_y):
            with tf.name_scope("accuracy"):
                is_correct = tf.to_float(tf.equal(true_y, pre_y))
                return tf.reduce_mean(is_correct)

        with tf.name_scope("metrics"):
            labels = self.targets
            predictions = self.predictions
            # 要求shape形状一致
            predictions.get_shape().assert_is_compatible_with(labels.get_shape())
            # 要求数据类型一致，不一致进行转换
            if labels.dtype != predictions.dtype:
                predictions = tf.cast(predictions, labels.dtype)
            # 基于预测索引id和实际的索引id，构建这个准确率
            accuracy_ = accuracy(true_y=labels, pre_y=predictions)
            tf.summary.scalar('accuracy', accuracy_)

            metrics = Metrics(accuracy=accuracy_, recall=None, f1=None)
        return metrics

    def restore(self, checkpoint_dir, session):
        """
        进行模型参数恢复操作(直接恢复)
        :param checkpoint_dir:
        :param session:
        :return:
        """
        # 0. 相关参数初始化
        if self.saver is None:
            self.saver = tf.train.Saver(**self.saver_parameters)

        # 1. 检查是否存在持久化的模型文件
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # 2. 进行判断
        if ckpt and ckpt.model_checkpoint_path:
            tf.logging.info("开始进行模型恢复操作:{}".format(ckpt.model_checkpoint_path))
            # 参数恢复
            self.saver.restore(sess=session, save_path=ckpt.model_checkpoint_path)
            # 恢复模型管理(保存磁盘中最多存在max_to_keep个模型)
            self.saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
        else:
            tf.logging.warn("从文件夹【{}】没有发现训练好的模型文件，不能进行模型恢复操作!!".format(checkpoint_dir))

    def save(self, session, save_path):
        # 0. 相关参数初始化
        if self.saver is None:
            self.saver = tf.train.Saver(**self.saver_parameters)

        # 1. 模型持久化
        tf.logging.info("进行模型持久化操作, 持久化路径为:{}".format(save_path))
        self.saver.save(sess=session, save_path=save_path, global_step=self.global_step)
        tf.logging.info("模型持久化完成!!!")
