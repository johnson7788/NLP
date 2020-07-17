# -- encoding:utf-8 --

import os
import csv
import numpy as np
import tensorflow as tf

from utils import data_helpers
from utils.vocabulary_utils import VocabularyProcessorUtil

# 数据文件
tf.flags.DEFINE_string("positive_data_file",
                       "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file",
                       "Data source for the positive data.")
# Eval Parameters
tf.flags.DEFINE_string("network_name", None, "给定模型名称!!!")
tf.flags.DEFINE_string("checkpoint_dir", "./model", "给定模型持久化的文件夹路径！")
tf.flags.DEFINE_string("vocab_model_path", "./model/vocab.pkl", "给定词汇模型所在的磁盘路径")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

FLAGS = tf.flags.FLAGS


def main(_):
    network_name = FLAGS.network_name
    if network_name is None:
        raise Exception("参数network_name必须给定!!!!")

    # 0. 数据校验，要求训练数据文件文件存在
    if not (os.path.isfile(FLAGS.positive_data_file) and os.path.isfile(FLAGS.negative_data_file)):
        raise Exception("给定的训练数据必须是文件路径的形成!!!")

    with tf.Graph().as_default():
        graph = tf.get_default_graph()
        with tf.Session() as sess:
            # 1. 加载词汇转换模型
            vocab_model_path = FLAGS.vocab_model_path
            if not tf.gfile.Exists(vocab_model_path):
                raise Exception("词汇转换模型必须存在,请检查磁盘路径:{}".format(vocab_model_path))
            vocab_model = VocabularyProcessorUtil.load_model(save_path=vocab_model_path)

            # 2. 恢复加载网络
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
            if not (ckpt and ckpt.model_checkpoint_path):
                raise Exception("不存在对应的模型文件，请检查:{}".format(FLAGS.checkpoint_dir))
            tf.logging.info("恢复模型:{}".format(ckpt.model_checkpoint_path))
            saver = tf.train.import_meta_graph("{}.meta".format(ckpt.model_checkpoint_path))
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)

            # 3. 获取Tensor对象
            inputs = graph.get_tensor_by_name("{}/placeholders/input_word_id:0".format(network_name.upper()))
            dropout_keep_prob = graph.get_tensor_by_name("{}/placeholders/dropout_keep_prob:0".format(network_name.upper()))
            predictions = graph.get_tensor_by_name("{}/project/predictions:0".format(network_name.upper()))

            # 4. 加载数据
            tf.logging.info("开始加载文本数据，并转换处理......")
            old_texts, labels = data_helpers.load_data_and_labels(
                positive_data_file=FLAGS.positive_data_file,
                negative_data_file=FLAGS.negative_data_file
            )

            # 4a. 文本数据id转换（截取、填充）
            texts = np.asarray(list(vocab_model.transform(old_texts)))
            # 4c. 构建批次
            batches = data_helpers.batch_iter(
                data=list(texts),
                batch_size=FLAGS.batch_size,  # 每个批次的样本数据量
                num_epochs=1,  # 总共迭代多少个epoch数据
                shuffle=False
            )

            # 5. 遍历数据进行预测
            all_predictions = []
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {inputs: x_test_batch, dropout_keep_prob: 1.0})
                # 数组拼接
                all_predictions = np.concatenate([all_predictions, batch_predictions])
            # 类型转换，以及格式/数据类型转换
            all_predictions = np.asarray(all_predictions, dtype=np.int32).reshape(-1)

            # 6. 效果评估
            correct_predictions = float(sum(all_predictions == labels))
            print("Total number of test examples: {}".format(len(labels)))
            print("Accuracy: {:g}".format(correct_predictions / float(len(labels))))
            print("实际值为:\n{}".format(labels))
            print("预测值为:\n{}".format(all_predictions))

            # 将评价保存到CSV
            predictions_human_readable = np.column_stack((all_predictions, labels, np.array(old_texts)))
            out_path = os.path.join(FLAGS.checkpoint_dir, "prediction.csv")
            print("Saving evaluation to {0}".format(out_path))
            # 参数：newline=''是给定不添加新行
            with open(out_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)  # 获取输出对象
                writer.writerows(predictions_human_readable)  # 输出CSV格式


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
