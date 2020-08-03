import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import data_utils
from tqdm import tqdm

class TextCNN(nn.Module):
    """
    TextCNN模型
    """
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        # 单词表的长度，用于做词嵌入lookingup查表
        vocab_num = args.vocab_num
        # 词嵌入后生成的单词的维度
        embed_dim = args.embed_dim
        # 类别的维度
        class_num = args.class_num
        # 卷积核的起始输入维度，因为开始只有1个维度输入，所以默认in_channels维度是1
        kernel_in = 1
        # 卷积核的数量，等于输出的卷积核的channel数量
        kernel_num = args.kernel_num
        #卷积核尺寸，是一个列表[3,4,5]
        kernel_sizes = args.kernel_sizes
        #单词做lookingup查表的词嵌入，将词id变成词向量
        self.embed = nn.Embedding(vocab_num, embed_dim)
        #ModuleList，子模型作为一个列表传入, kernel_size卷积核的尺寸，这里是分别是[3,embed_dim], [4,embed_dim], [5,embed_dim]
        #kernel_size卷积核的尺寸的形状是[H,W], 高是3，代表3个词之间的关系
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(in_channels=kernel_in, out_channels=kernel_num, kernel_size=(size, embed_dim))
             for size in kernel_sizes]
        )
        #做一次dropout
        self.dropout = nn.Dropout(args.dropout)
        #做全连接, 输入维度是len(kernel_sizes) * kernel_num， 因为是把所有卷积后的结果进行拼接，所以这个是拼接后的维度，class_num是要预测的类别
        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, class_num)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, text):
        """
        前向网络
        :param text: text的维度是[batch_size, sequence_length], 输入的是text的单词的id
        :return:
        """
        # 对text进行embedding lookup，生成的维度是[batch_size,sequence_length,Embedding_demission], 即[N,W,D]
        x = self.embed(text)
        # 添加一个维度，用于卷积，在第二个维度上扩充，变成[batch_size,1,sequence_length,Embedding_demission],
        x = x.unsqueeze(1)
        # 使用ModuleList中的卷积，卷积后进行relu激活，激活后,
        #第一次卷积Conv2d(1, 100, kernel_size=(3, 128), stride=(1, 1))，输入的x[batch_size,1,sequence_length,Embedding_demission], 卷积后x[batch_size,kernel_num,sequence_length,1], squeeze最后一个维度
        #第二次Conv2d(1, 100, kernel_size=(4, 128), stride=(1, 1)), 输出的形状和第一次相同
        #第三次Conv2d(1, 100, kernel_size=(5, 128), stride=(1, 1))， 输出的形状和第一次相同
        x_conv_pool_result = []
        #分别进行3次卷积，x_conv_result存储3次卷积的结果, 分布进行池化操作
        for conv in self.convs1:
            #输入的x[batch_size,1,sequence_length,Embedding_demission], 卷积后x[batch_size,kernel_num,sequence_length,1]
            x1 = conv(x)
            #激活不改变形状
            x1 = F.relu(x1)
            #squeeze后 [batch_size, kernel_num, sequence_length】
            x1 = x1.squeeze(3)
            #x1的shape是[batch_size,kernel_num,sequence_length], 设置kernel_size的大小是sequence_length * sequence_length
            x1 = F.max_pool1d(x1, kernel_size=x1.size(2))
            # max_pool1d后输出的x1的shape是[batch_size, kernel_num, 1]
            x1 = x1.squeeze(2)
            #squeeze后的shape是[batch_size, kernel_num]
            x_conv_pool_result.append(x1)
        #拼接输出结果, 形状是[batch_size, kernel_num*卷积的次数]
        x = torch.cat(x_conv_pool_result, 1)
        #做一次dropout, 形状不变
        x = self.dropout(x)
        #做全连接后得到输出结果 [batch_size, class_num]
        logit = self.fc1(x)
        return logit


def train(train_iter, model, args):
    """
    训练
    :param train_iter: 训练数据
    :param model:  模型，例如初始化的TextCNN
    :param args: paraser传入的config信息
    :return:
    """
    print("开始训练模型")
    #创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    #如果有GPU，使用gpu
    if args.cuda:
        model.cuda()

    for epoch in range(1, args.epochs+1):
        training_loss = 0.0
        training_acc = 0.0
        training_count = 0.0

        for batch in tqdm(train_iter):
            # batch.text返回的形状是[sequence_length, batch_size], batch.label[batch_size]
            feature, target = batch.text, batch.label
            #feautre进行转置，形状变成【batch_size, sequence_length]
            feature.t_()
            # 所有label的数值减去1
            target.sub_(1)
            #如果是gpu，转换成gpu资源
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            #得到预测结果
            logit = model(feature)
            #计算交叉熵损失
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            #计算准确率
            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            #损失training_loss更新
            training_loss += loss.item()
            training_acc += corrects.item()
            training_count += batch.batch_size

        #计算平均算是和准确率
        training_loss /= training_count
        training_acc /= training_count
        accuracy = 100.0 * training_acc
        print('Training epoch [{}/{}] - loss: {:.6f}  acc: {:.2f}%'.format(
            epoch, args.epochs, training_loss, accuracy))
        #保存模型
        if epoch % args.save_interval == 0:
            torch.save(model, args.save_path + f"textcnn.model-{epoch}")
            print('保存模型完成')
    #训练完成后再次保存模型
    torch.save(model, args.save_path + "textcnn.model")
    print("训练完成")


def eval(data_iter, model, args):
    """
    评估模型
    :param train_iter: 训练数据
    :param model:  模型，例如初始化的TextCNN
    :param args: paraser传入的config信息
    :return:
    """
    print("开始评估模型")
    #设置评估模型
    model.eval()
    if args.cuda:
        model.cuda()
    #评估准确率和损失
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_()
        target.sub_(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data.item()
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%'.format(avg_loss, accuracy))
    print('评估完成')
    return accuracy

def predict(path, model, text_field, label_feild, cuda):
    """
    模型预测
    :param path: 要预测文本文件的路径
    :param model: 初始化好的模型
    :param text_field: text_field 文件
    :param label_feild:
    :param cuda: 是否使用gpu
    :return:
    """
    model.eval()
    if cuda:
        model.cuda()

    document = ''
    with open(path, encoding="utf8", errors='ignore') as f:
        for line in f:
            if line != '\n':
                document += data_utils.text_filter(line)

    #对文本进行jieba处理
    text = text_field.preprocess(document)

    #文本转换成id
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.LongTensor(text)
    if cuda:
        x = x.cuda()
    #预测结果
    output = model(x)
    #获取概率最大的结果
    _, predicted = torch.max(output, 1)
    #预测的索引id转换成文字
    label = label_feild.vocab.itos[predicted.data[0] + 1]
    return document, label
