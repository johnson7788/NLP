import torch

class Config(object):
    """Base configuration class."""
    #指定包含训练集，验证集合测试集的文件夹
    data_directory = "data/zhengjian/"
    #模型的保存位置
    save_path='model/'
    #模型保存名称
    save_name='checkpoint.pth'
    #使用哪个模型类型，可选 ['densenet161', 'resnet18', 'vgg16', 'res2next50']
    arch = 'res2next50'
    # classifier的 隐藏层数, 可以任意个[1024,512,256]，每个是一个FC
    hidden_units = [256]
    # 评估间隔, 训练多少个epoch，进行一次评估
    eval_interval = 100
    # 是否绘图还是直接返回结果
    plot = False
    # 绘图显示的预测个数, 需要是偶数个
    plot_image = 6
    #是否使用gpu
    cuda = False
    #device name ,如果使用cpu，那么就是cpu，如果使用gpu, 可能是第几块显卡cuda:0
    device_name = 'cpu'
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
