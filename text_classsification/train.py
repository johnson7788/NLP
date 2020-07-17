# -- encoding:utf-8 --

import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from nets.text_cnn import Network as TextCNN
from nets.text_rnn import Network as TextRNN
from nets.text_rnn_improve import Network as TextRNNImprove
from nets.text_rnn_improve2 import Network as TextRNNImprove2
from nets.text_transformer import Network as TextTransformer
from nets.text_rnn_transformer import Network as TextRNNTransformer
from nets.text_cnn_transformer import Network as TextCNNTransformer
from nets.text_cnn_rnn import Network as TextCNNRNN
from nets.text_adversarial_rnn_improve import Network as TextAdversarialRNN
from utils.vocabulary_utils import VocabularyProcessorUtil
from utils import data_helpers
from utils import network_utils

# parameters
# ===================================================
# 训练数据存储路径相关参数
tf.flags.DEFINE_string("positive_data_file", "Data source for the positive data")
tf.flags.DEFINE_string("negative_data_file", "Data source for the negative data")
tf.flags.DEFINE_float("dev_sample_percentage", 0.05, "验证数所占的比例值!!")

# ===================================================
# 模型整体相关参数
tf.flags.DEFINE_string("model", "text_cnn", "给定具体使用什么模型，可选值为:[text_cnn, text_rnn]")
tf.flags.DEFINE_string("network_name", None, "给定模型名称, 如果没有给定，使用参数model(大写)字符串")
tf.flags.DEFINE_integer("num_class", 2, "给定文本分类的类别数目!!!")

# ===================================================
# 训练相关参数
tf.flags.DEFINE_integer("batch_size", 16, "一个训练批次中的样本数目!!!")
tf.flags.DEFINE_integer("max_epochs", 200, "给定训练多少轮数据!!!")
tf.flags.DEFINE_float("dropout_keep_prob", 0.9, "Dropout的时候保留的百分比!!")

# ===================================================
# TextCNN参数
tf.flags.DEFINE_string("num_filters", "128",
                       "给定卷积核的数量，使用逗号分割开的int数据，int数据个数要不是1个要不和region_sizes的大小一致, eg:128,256,128")
tf.flags.DEFINE_string("region_sizes", "2,3,4", "给定卷积的时候，提取单词特征的范围大小。使用逗号分割开的int类型数据")

# ===================================================
# TextRNN参数
tf.flags.DEFINE_integer("num_units", 128, "给定RNN Cell中的神经元数目!!!")
tf.flags.DEFINE_integer("layers", 3, "给定RNN的层次数目!!!")

# ===================================================
# Transformer参数
tf.flags.DEFINE_integer("attention_dimension_size", 128, "给定Self-Attention的输出维度大小!!!")
tf.flags.DEFINE_integer("attention_layers", 3, "给定Transformer中Encoder Attention的层次数目!!!")
tf.flags.DEFINE_integer("attention_headers", 3, "给定Attention中header的数目!!!")

# ===================================================
# Embedding相关参数
tf.flags.DEFINE_integer("embedding_dimensions", 128, "给定单词词向量转换的时候过程中的向量维度大小!!")
tf.flags.DEFINE_boolean("with_word2vec", False, "是否使用Word2Vec初始化Embedding Table值！！")
tf.flags.DEFINE_string("word2vec_model_path", "./model/w2v.bin", "给定Word2Vev的模型存储磁盘路径！！")
tf.flags.DEFINE_bool("train_embedding_table", False,
                     "当使用Word2Vec的初始化向量的时候，是否训练Embedding Table的参数值!!!")

# ===================================================
# 优化器相关参数
tf.flags.DEFINE_float("l2_weight_decay", 0.01, "l2正则惩罚项的系数!!")
tf.flags.DEFINE_string("optimizer_type", "adam",
                       "给定优化器的类别，可选范围为:[adam, sgd, adadelta, adagrad, ftrl, momentum]")
tf.flags.DEFINE_float("adam_beta1", 0.9, "Adam优化器参数！！")
tf.flags.DEFINE_float("adam_beta2", 0.999, "Adam优化器参数！！")
tf.flags.DEFINE_float("adam_epsilon", 1e-8, "Adam优化器参数！！")
tf.flags.DEFINE_float("momentum", 0.01, "Momentum优化器参数！！")
tf.flags.DEFINE_string("learning_rate_type", "exponential",
                       "给定学习率的变化方式，可选值为:[constant, polynomial, exponential]")
tf.flags.DEFINE_float("base_learning_rate", 0.01, "给定的基础学习率!!!")
tf.flags.DEFINE_float("lr_decay_steps", 100, "给定学习率的缩放间隔批次大小!!")
tf.flags.DEFINE_float("lr_decay_rate", 0.9, "给定学习率的缩放比例大小!!")
tf.flags.DEFINE_boolean("lr_staircase", False, "给定学习率进行指数缩放的时候，是否进行整间隔的缩放!!!")
tf.flags.DEFINE_float("end_learning_rate", 1e-5, "给定学习率进行多项式缩放的时候，最终的学习率大小!!!")

# ====================================================
# 模型持久化相关参数
tf.flags.DEFINE_string("checkpoint_dir", "./model", "给定模型持久化的文件夹路径！")
tf.flags.DEFINE_string("vocab_model_path", "./model/vocab.pkl", "给定词汇模型所在的磁盘路径")
tf.flags.DEFINE_integer("checkpoint_per_batch", 50, "给定模型持久化的间隔批次大小!")
tf.flags.DEFINE_integer("max_to_keep", 2, "给定最多持久化的模型版本数目！")

# ====================================================
# 模型可视化相关参数
tf.flags.DEFINE_string("summary_dir", "./graph", "模型可视化文件数据保存的文件夹路径！")
tf.flags.DEFINE_integer("evaluate_per_batch", 100, "给定每间隔多少批次进行一次数据效果验证!!!")

# ===================================================
# Session相关参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "当GPU上不支持当前操作的时候，是否允许自动调整为CPU运行！")
tf.flags.DEFINE_boolean("log_device_placement", False, "是否打印日志!")

# ===================================================
FLAGS = tf.flags.FLAGS


def check_directory(path, create=True):
    flag = os.path.exists(path)
    if not flag:
        if create:
            os.makedirs(path)
            flag = True
    return flag


def load_model_class(name):
    name2class = {
        "text_cnn": TextCNN,
        "text_rnn": TextRNN,
        "text_rnn_improve": TextRNNImprove,
        "text_rnn_improve2": TextRNNImprove2,
        "text_transformer": TextTransformer,
        "text_rnn_transformer": TextRNNTransformer,
        "text_cnn_rnn": TextCNNRNN,
        "text_cnn_transformer": TextCNNTransformer,
        "text_adversarial_rnn": TextAdversarialRNN
    }
    if name not in name2class:
        raise Exception("参数一次，模型名称不存在，请选择【{}】其中的任意一个模型!!".format(name2class.keys()))
    return name2class[name]


def main(_):
    """
    主体函数代码构建
    :param _:
    :return:
    """
    # -1. 模型持久化文件夹检查
    check_directory(FLAGS.checkpoint_dir)
    checkpoint_save_path = os.path.join(FLAGS.checkpoint_dir, "model.ckpt")
    # 0. 数据校验，要求训练数据文件文件存在
    if not (os.path.isfile(FLAGS.positive_data_file) and os.path.isfile(FLAGS.negative_data_file)):
        raise Exception("给定的训练数据必须是文件路径的形成!!!")

    # 1. 加载词汇转换模型
    vocab_model_path = FLAGS.vocab_model_path
    if not tf.gfile.Exists(vocab_model_path):
        raise Exception("词汇转换模型必须存在,请检查磁盘路径:{}".format(vocab_model_path))
    vocab_model = VocabularyProcessorUtil.load_model(save_path=vocab_model_path)

    # 2. TensorFlow相关代码构建
    with tf.Graph().as_default():
        # 一、执行图的构建
        # 1. 网络构建(前向执行构成的构建)
        tf.logging.info("开始构建前向过程的网络结构....")
        # 获取使用什么模型以及模型的名称
        model_name = FLAGS.model.lower()
        model_class = load_model_class(model_name)
        network_name = model_name.upper() if FLAGS.network_name is None else FLAGS.network_name
        # 构建Embedding Lookup相关参数
        with_word2vec = False
        embedding_table = None
        if FLAGS.with_word2vec:
            if os.path.exists(FLAGS.word2vec_model_path):
                tf.logging.info("加载Word2Vec训练好的词向量转换模型!!!")
                embedding_table, _ = VocabularyProcessorUtil.load_word2vec_embedding(
                    save_path=FLAGS.word2vec_model_path)
                with_word2vec = True
            else:
                tf.logging.warn("不能加载Word2Vec词向量转换矩阵，原因是文件不存在，请检查:{}".format(FLAGS.word2vec_model_path))
        # 构建TextCNN的卷积相关参数(卷积核数量、卷积的范围)
        num_filters = list(map(int, FLAGS.num_filters.split(",")))
        num_filters = num_filters if len(num_filters) > 1 else num_filters[0]
        region_sizes = list(map(int, FLAGS.region_sizes.split(",")))
        # 构建模型
        model = model_class(
            with_word2vec=with_word2vec,  # 是否做Word2Vec
            vocab_size=len(vocab_model.vocabulary_),  # 词汇数目
            embedding_dimensions=FLAGS.embedding_dimensions,  # Embedding Loopup转换的向量维度大小
            embedding_table=embedding_table,  # 做Word2Vec的初始化向量矩阵
            train_embedding_table=FLAGS.train_embedding_table,  # 是否进行Embedding Table的训练
            num_class=FLAGS.num_class,  # 类别数目
            network_name=network_name,  # 网络名称
            weight_decay=FLAGS.l2_weight_decay,  # 正则惩罚项系数
            optimizer_type=FLAGS.optimizer_type,  # 优化器类别
            optimizer_parameters_func=network_utils.build_optimizer_parameters_func(flags=FLAGS),
            saver_parameters={'max_to_keep': FLAGS.max_to_keep},
            num_units=FLAGS.num_units,  # RNN的神经元数目
            layers=FLAGS.layers,  # RNN的层次数目
            num_filters=num_filters,  # TextCNN卷积核的数目
            region_sizes=region_sizes,  # TextCNN中卷积提取特征的时候单词的范围大小
            attention_dimension_size=FLAGS.attention_dimension_size,  # Self-Attention的输出维度大小
            attention_layers=FLAGS.attention_layers,  # Transformer中Encoder Attention的层次数目
            attention_headers=FLAGS.attention_headers,  # Attention中header的数目
        )

        # 2. 计算损失函数
        tf.logging.info("开始构建损失函数对象....")
        total_loss = model.losses()
        # 3. 构建优化器以及训练操作对象
        tf.logging.info("开始构建模型训练操作对象.....")
        _, train_op = model.optimizer(loss=total_loss)
        # 4. 模型评估指标的构建
        tf.logging.info("开始构建模型评估指标.....")
        metrics = model.metrics()
        # 5. 模型可视化相关操作
        tf.logging.info("开始构建模型可视化相关信息.....")
        summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(FLAGS.summary_dir, "train")
        eval_summary_dir = os.path.join(FLAGS.summary_dir, "eval")
        check_directory(train_summary_dir)
        check_directory(eval_summary_dir)
        train_summary_writer = tf.summary.FileWriter(logdir=train_summary_dir, graph=tf.get_default_graph())
        eval_summary_writer = tf.summary.FileWriter(logdir=eval_summary_dir, graph=tf.get_default_graph())

        # 二、执行图的运行
        session_config = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        with tf.Session(config=session_config) as sess:
            # 1. 模型参数恢复(先初始化所有模型参数，然后再进行模型参数恢复)
            tf.logging.info("进行模型参数初始化相关操作......")
            sess.run(tf.global_variables_initializer())  # 参数随机初始化
            model.restore(checkpoint_dir=FLAGS.checkpoint_dir, session=sess)  # 参数恢复

            # 2. 加载数据
            tf.logging.info("开始加载文本数据，并转换处理......")
            texts, labels = data_helpers.load_data_and_labels(
                positive_data_file=FLAGS.positive_data_file,
                negative_data_file=FLAGS.negative_data_file
            )
            # 2a. 文本数据id转换（截取、填充,只保留512位）
            texts = np.asarray(list(vocab_model.transform(texts)))
            # 2b. 将数据进行分割，构建训练数据和验证数据
            x_train, x_eval, y_train, y_eval = train_test_split(texts, labels,
                                                                test_size=FLAGS.dev_sample_percentage, random_state=28)
            tf.logging.info("训练数据格式:{}---{}".format(np.shape(x_train), np.shape(y_train)))
            tf.logging.info("验证数据格式:{}---{}".format(np.shape(x_eval), np.shape(y_eval)))
            # 2c. 将数据做一个封装转换为批次迭代器
            batches = data_helpers.batch_iter(
                data=list(zip(x_train, y_train)),
                batch_size=FLAGS.batch_size,  # 每个批次的样本数据量
                num_epochs=FLAGS.max_epochs  # 总共迭代多少个epoch数据
            )

            def train_step(_x, _y, writer):
                feed_dict = {
                    model.inputs: _x,
                    model.targets: _y,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                step, _, _accuracy, _loss, _summary = sess.run(
                    [model.global_step, train_op, metrics.accuracy, total_loss, summary_op],
                    feed_dict=feed_dict)
                time_str = datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, _loss, _accuracy))
                writer.add_summary(_summary, step)
                return step

            def dev_step(_x, _y, writer=None):
                feed_dict = {
                    model.inputs: _x,
                    model.targets: _y
                }
                step, _accuracy, _loss, _summary = sess.run(
                    [model.global_step, metrics.accuracy, total_loss, summary_op], feed_dict=feed_dict)
                time_str = datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, _loss, _accuracy))
                if writer is not None:
                    writer.add_summary(_summary, global_step=dev_summary_global_step)

            # 3. 遍历数据进行模型训练、模型持久化
            for batch in batches:
                # a. 从batch中提取x和y
                x_batch, y_batch = zip(*batch)
                # b. 模型训练
                current_step = train_step(x_batch, y_batch, writer=train_summary_writer)
                # c. 每隔一定的训练间隔，进行验证数据的效果评估
                if current_step % FLAGS.evaluate_per_batch == 0:
                    print("\nEvaluation:")
                    dev_summary_global_step = current_step
                    dev_batches = data_helpers.batch_iter(
                        data=list(zip(x_eval, y_eval)),
                        batch_size=FLAGS.batch_size * 10,  # 每个批次的样本数据量
                        num_epochs=1  # 总共迭代多少个epoch数据
                    )
                    for dev_batch in dev_batches:
                        dev_x_batch, dev_y_batch = zip(*dev_batch)
                        dev_step(dev_x_batch, dev_y_batch, writer=eval_summary_writer)
                        dev_summary_global_step += 1
                # d. 每个一定的训练间隔，进行模型持久化
                if current_step % FLAGS.checkpoint_per_batch == 0:
                    model.save(session=sess, save_path=checkpoint_save_path)

            # 最终结束的时候再保存一次模型
            model.save(session=sess, save_path=checkpoint_save_path)


if __name__ == '__main__':
    # 1. 设置一下TensorFlow的日志级别
    tf.logging.set_verbosity(tf.logging.DEBUG)

    # 2. 开始代码运行，默认调用当前py文件中的main函数
    tf.app.run()
