import torch

class Config(object):
    """Base configuration class."""
    #训练文件夹位置
    train_dir = "data/train"
    #评估文件夹位置
    eval_dir = "data/eval"
    #模型的保存位置
    save_path='model/'
    #是否使用gpu
    cuda = True
    #训练的epoch
    epochs = 2
    batch_size = 64
    #学习率
    learning_rate = 0.001
    #学习率动量
    learning_momentum = 0.9
    #学习率衰减稀疏
    weight_decay = 0.0001
    dropout = 0.5
    #生成的词嵌入的维度
    embed_dim = 128
    #卷积核的数量
    kernel_num = 100
    #卷积核的尺寸
    kernel_sizes = "3,4,5"
    #训练多少个epoch时，模型保存
    save_interval = 2

    #初始化，是否使用gpu
    def __init__(self):
        if self.cuda:
            self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.cuda else "cpu")

    def dump(self):
        """打印配置信息"""
        print("模型配置如下:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("\t{:30} = {}".format(a, getattr(self, a)))
        print()
