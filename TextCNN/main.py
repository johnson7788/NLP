import os, sys
import argparse
import torch
import model_utils
import data_utils
from config import Config

#TextCNN模型
def dotrain():
    parser = argparse.ArgumentParser(description='Text CNN 分类器')
    parser.add_argument('--model', type=str, default="model/textcnn.model", help='读取model继续训练')
    conf = Config()
    #打印模型配置信息
    conf.dump()
    args = parser.parse_args()
    if not os.path.isdir("model"):
        os.mkdir("model")
    print("处理训练数据")
    train_iter, text_field, label_field = data_utils.text_dataloader(conf.train_dir, conf.batch_size)
    #使用pickle保存字典到本地
    data_utils.save_vocab(text_field.vocab, "model/text.vocab")
    data_utils.save_vocab(label_field.vocab, "model/label.vocab")

    #添加新的配置，嵌入的维度vocab_num， 分类的类别数量class_num，
    conf.vocab_num = len(text_field.vocab)
    conf.class_num = len(label_field.vocab) - 1
    # 卷积核大小, 代表跨越的句子和字的大小, 找打相邻字直接的联系, 例如[3, 4, 5]
    conf.kernel_sizes = [int(k) for k in conf.kernel_sizes.split(',')]

    #模型加载和初始化
    if os.path.exists(args.model):
        print('发现模型文件, 加载模型: {}'.format(args.model))
        cnn = torch.load(args.model)
    else:
        cnn = model_utils.TextCNN(conf)
    #模型训练
    try:
        model_utils.train(train_iter, cnn, conf)
    except KeyboardInterrupt:
        print('-' * 80)
        print('提前退出训练.')

#评估模型
def doeval():
    parser = argparse.ArgumentParser(description='Text CNN 分类器')
    #必须指定已经训练好的模型
    parser.add_argument('--model', type=str, default="model/textcnn.model", help='读取model进行评估')
    conf = Config()
    #打印模型配置信息
    conf.dump()
    args = parser.parse_args()
    print("加载测试数据")
    #测试时不进行数据打乱操作
    eval_iter, text_field, label_field = data_utils.text_dataloader(conf.eval_dir, conf.batch_size, shuffle=False)
    # 模型加载和初始化
    if os.path.exists(args.model):
        print('发现模型文件, 加载模型: {}'.format(args.model))
        cnn = torch.load(args.model)
    else:
        print("未找到模型文件，退出")
        sys.exit(-1)
    #加载以保存的字典
    text_field.vocab = data_utils.load_vocab("model/text.vocab")
    label_field.vocab = data_utils.load_vocab("model/label.vocab")
    #开始模型评估
    model_utils.eval(eval_iter, cnn, conf)

#预测
def dopredict():
    """
    给定一个文件或一句话，预测结果
    :return:
    """
    parser = argparse.ArgumentParser(description='Text CNN 分类器')
    #必须指定已经训练好的模型
    parser.add_argument('--path', type=str, default="data/predict/",help='要进行预测的文本文件的路径,或文件夹')
    parser.add_argument('--model', type=str, default="model/textcnn.model", help='读取model进行预测')
    conf = Config()
    args = parser.parse_args()
    #指定Field格式
    text_field = data_utils.TextTEXT
    label_field = data_utils.TextLABEL
    text_field.vocab = data_utils.load_vocab("model/text.vocab")
    label_field.vocab = data_utils.load_vocab("model/label.vocab")
    # 模型加载和初始化
    if os.path.exists(args.model):
        print('发现模型文件, 加载模型: {}'.format(args.model))
        cnn = torch.load(args.model)
    else:
        print("未找到模型文件，退出")
        sys.exit(-1)
    #如果是文件夹，那么预测里面的文件，否则就是文件，直接预测
    if os.path.isdir(args.path):
        files = os.listdir(args.path)
        files_path = [args.path+f for f in files]
    else:
        files_path = [args.path]
    #开始预测
    for file in files_path:
        text, label = model_utils.predict(file, cnn, text_field, label_field, conf.cuda)
        print('[path]  {}\n[Text]  {}\n[Label] {}\n'.format(file, text, label))
    print(f'共预测{len(files_path)}个文件')

if __name__ == '__main__':
    dotrain()
    # doeval()
    # dopredict()