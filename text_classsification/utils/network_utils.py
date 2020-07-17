# -- encoding:utf-8 --

import tensorflow as tf


def build_optimizer_parameters_func(flags):
    """
    构建优化器的参数
    :param flags:
    :return:
    """
    optimizer_type = flags.optimizer_type
    parameters = {}

    # 添加各自不同优化器对应的参数
    if optimizer_type == 'adam':
        parameters['beta1'] = flags.adam_beta1
        parameters['beta2'] = flags.adam_beta2
        parameters['epsilon'] = flags.adam_epsilon
    elif optimizer_type == 'momentum':
        parameters['momentum'] = flags.momentum

    def build_optimizer_parameters(global_step):
        # 添加共同参数: learning_rate
        learning_rate_type = flags.learning_rate_type
        base_learning_rate = flags.base_learning_rate
        if learning_rate_type == 'exponential':
            tf.logging.info("使用指数变化学习率形式.....")
            # staircase=False：decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
            # staircase=True：decayed_learning_rate = learning_rate * decay_rate ^ int(global_step / decay_steps)
            lr = tf.train.exponential_decay(
                learning_rate=base_learning_rate,  # 基础学习率
                global_step=global_step,  # 迭代的步数
                decay_steps=flags.lr_decay_steps,  # 间隔大小
                decay_rate=flags.lr_decay_rate,  # 缩放比例
                staircase=flags.lr_staircase,  # 是否整间隔的进行缩放
                name="exponential_learning_rate")
            pass
        elif learning_rate_type == 'polynomial':
            tf.logging.info("使用多项式变化学习率形式.....")
            # global_step = min(global_step, decay_steps)
            # decayed_learning_rate = (learning_rate - end_learning_rate) * (1 - global_step / decay_steps) ^ (power) + end_learning_rate
            lr = tf.train.polynomial_decay(
                learning_rate=base_learning_rate,  # 基础学习率
                global_step=global_step,  # 迭代的步数
                decay_steps=flags.lr_decay_steps,  # 间隔大小
                end_learning_rate=flags.end_learning_rate,  # 最终学习率大小
                power=1.0,  # 给定是否的时候是否是线性的系数
                cycle=True,  # 当学习率为最小值的时候，是否将学习率重置设置比较大，然后再进行学习率下降的操作
                name="polynomial_learning_rate")
        else:
            tf.logging.info("使用常数不变的学习率.....")
            lr = tf.constant(base_learning_rate, name='lr')
        parameters['learning_rate'] = lr
        tf.summary.scalar('learning_rate', lr)

        return parameters

    return build_optimizer_parameters
