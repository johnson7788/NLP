import matplotlib.pyplot as plt
import matplotlib
import time
import copy
import numpy as np
import os
import errno
import sys
from progress.bar import ShadyBar
from tqdm import tqdm

from data_utils import prediction_class_names

# Import Pytorch modules
import torch
from torch import optim, nn
from torchvision import models


def classify_image(image_tensor, model, top_k, gpu):
    """分类图片，用于预测阶段

    Args:
        image_tensor (torch.FloatTensor): 处理好的图片Tensor作为input
        model (): 训练好的额模型
        gpu (bool): 是否使用gpu

    Returns:
        probs ([float]): 预测的置信度列表
        preds ([int]): 预测的类别列表
    """
    device = select_device(gpu=gpu)
    print(f'\n使用device {device} 进行预测')
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = torch.exp(model(image_tensor))
    probs, preds = output.topk(top_k, dim=1)
    probs, preds = probs.tolist()[0], preds.tolist()[0]

    # 将对应的class索引映射为类别名称
    preds = prediction_class_names(
            predictions=preds,
            class_to_idx=model.class_to_idx)
    return probs, preds


def create_model(arch='vgg16', class_to_idx=None, hidden_units=[], drop_p=0.5, device_name='cpu'):
    """
    根据选择不同的预训练模型创建模型，使用torchvision的预训练模型，并且定义自己的多个全连接层作和最后一层
    Args:
        arch (str): 预训练模型的名称, 支持的模型:  ['densenet161', 'resnet18', 'vgg16', 'res2next50']
        hidden_units ([int]): 隐藏层神经元大小, 自定义的classifier可以是多个全连接层
        drop_p (float (0, 1)): dropout值
        class_to_idx (dict): 类别到索引

    Returns:
        model (nn model): 返回一个自定义了classifier层的pre-trained模型
                          Attributes:
                              arch (str)
                              class_to_idx (dict)
                              output_size (int)
                              hidden_layers ([int])
                              drop_p (float)
    """
    print('\n开始创建模型')
    # 支持模型列表,keys = 模型名字，values = 自定义的classifier输入尺寸，根据模型的最后一个卷积的输出尺寸确定, res2next50: 2048
    supported_models = {'densenet161': 2208,
                        'resnet18': 512,
                        'vgg16': 25088,
                        'res2next50': None,}
    #获取classifier的输入尺寸
    try:
        input_size = supported_models[arch]
    except KeyError:
        print(f'Exception: 模型架构 {arch} 是不支持的'
              f' 支持的模型架构是: {list(supported_models.keys())}.')
        sys.exit(1)

    # 定义classifier的输出大小，即输出的类别个数
    output_size = len(class_to_idx)

    #加载预训练模型
    if arch in ['res2next50'] :
        from res2next import res2next50
        model = res2next50(pretrained=True, map_location=device_name)
    else:
        model = getattr(models, arch)(pretrained=True)

    #冻结预训练模型参数
    for param in model.parameters():
        param.requires_grad = False

    #获取真实的input_size
    if not input_size and arch in ['res2next50'] :
        input_size= model.fc.in_features
    #创建自定义classifier
    #合并隐藏层和最后一层的classifier
    layer_sizes = [input_size]
    if hidden_units:
        layer_sizes += hidden_units
    layer_sizes += [output_size]

    #创建layers
    classifier = nn.Sequential()
    #layer_sizes[:-1] 和layer_sizes[1:]是全连接的输入和输出要对应
    for idx, (inp, out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        #添加一个全连接层fc0，fc1。。等
        classifier.add_module('fc' + str(idx), nn.Linear(inp, out))
        # 在FC层后添加ReLu和Dropout层除了最后一个FC层
        if idx < (len(layer_sizes) - 2):
            classifier.add_module('relu' + str(idx), nn.ReLU())
            classifier.add_module('dropout' + str(idx), nn.Dropout(drop_p))
    #返回logSoftmax结果
    classifier.add_module('output', nn.LogSoftmax(dim=1))

    # 针对不同的预训练模型，更改为不同的classifier，应为名字不同
    if arch in ['resnet18','res2next50']:
        model.fc = classifier
    else:
        model.classifier = classifier

    #为了以后方便引用，添加一些模型参数信息
    model.arch = arch
    model.class_to_idx = class_to_idx
    model.output_size = output_size
    model.hidden_units = hidden_units
    model.drop_p = drop_p

    print('\n模型创建完毕')
    print(model)

    return model


def create_optimizer(model, lr=0.001):
    """
    为模型的最后的Classifier层添加优化器
    使用Adam优化器，只是模型的最后一个 fc/classifier layer(s)的优化器，只优化这fc/classifier的参数
    Args:
        model (): 通过create_model()创建的模型类
        lr (float): 学习率

    Returns:
        优化器 (torch.optim.Optimizer)
    """
    print('\n开始创建优化器')

    #获取模型参数
    if model.arch in ['resnet18','res2next50']:
        params = model.fc.parameters()
    else:
        params = model.classifier.parameters()
    #创建优化器并返回
    optimizer = optim.Adam(params, lr=lr)
    return optimizer


def load_checkpoint(checkpoint_path, load_optimizer=False, gpu=False):
    """
    加载checkpoint，重建预训练模型，为了继续运行已经保存好的模型
    Args:
        checkpoint_path (str): checkpoint文件路径
        load_optmizer (bool): True: 创建一个优化器，并载入state_dict， 训练时使用
                              False: 不加载优化器，为了预测时使用
        gpu (bool): 是否使用GPU

    Returns:
        model (): 重建好的 pre-trained 模型
        optimizer (optim.Optimizer, None): 只有load_optimizer为True时返回优化器，否则返回None
        epoch (int): 效果最好的epoch是哪个
        history (dict): 训练的历史记录
    """

    print('\n开始加载模型的checkpoint')

    device = select_device(gpu)

    # 加载checkpoint，不同的设备，checkpoint也不同
    if gpu:
        checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    epoch = checkpoint['epoch']
    history = checkpoint['history']

    #加载模型
    model = create_model(arch=checkpoint['arch'],
                         class_to_idx=checkpoint['class_to_idx'],
                         hidden_units=checkpoint['hidden_units'],
                         drop_p=checkpoint['drop_p'])

    #加载模型权重
    print('\n开始记载模型权重状态')
    model.load_state_dict(checkpoint['model_state_dict'])

    #使用device
    model.to(device)

    #是否使用优化器
    if load_optimizer:
        optimizer = create_optimizer(model=model)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        optimizer = None

    print(f'\n成功加载第 {epoch} 个epoch的模型')
    return model, optimizer, epoch, history


def select_device(gpu):
    """
    选择CPU或GPU
    Args:
        gpu (bool): 选择GPU如果True，否则CPU

    Returns:
        device (torch.device): 'cpu' or 'cuda:0'
    """
    if gpu:
        assert torch.cuda.is_available(), ('Error: Requested GPU, '
                                           'but GPU is not available.')

    #选择device
    device = torch.device('cuda:0') if gpu else torch.device('cpu')
    return device


def train_model(dataloaders, model, optimizer, gpu=True,
                start_epoch=1, epochs=2, train_history=None):
    """
    开始训练模型，model.state_dict 和optimizer.state_dict保存着验证集最好时的效果
    Args:
        model (PyTorch model) 模型类
        dataloaders (dict): pytorch的dataloaders，字典格式keyes = ['train', 'eval', 'test'];values = Dataloaders
        criterion (): 训练集损失函数
        optimizer (optim. Optimizer) 优化器
        gpu (bool):  是否使用gpu训练
        start_epoch (int): 从第几个epoch开始
        epochs (int): 要训练多少个epoch
        train_history (dict): 训练集和验证集的损失和准确率历史记录
                             history = {
                                'train': {'loss': [], 'acc': []},
                                'eval': {'loss': [], 'acc': []}}
                            如果 start_epoch <= len(history['..']['..'][]),
                            历史记录会清除从start_epoch以后的记录

    Returns:
        history (dict(dict)): 嵌套的字典包含训练集和验证集的准确率和损失
        best_epoch (int): 验证集和训练集最好的epoch
    """
    #negative log likelihood loss 损失函数
    criterion = nn.NLLLoss()

    # 设置history和 best state, 如果不存在history，创建空的
    if train_history is None:
        history = {
            'train': {'loss': [], 'acc': []},
            'eval': {'loss': [], 'acc': []}
        }
    else:
        history = train_history
        # 清除start_epoch后面的历史记录
        history['train']['loss'] = history['train']['loss'][0: start_epoch - 1]
        history['eval']['loss'] = history['eval']['loss'][0: start_epoch - 1]
        history['train']['acc'] = history['train']['acc'][0: start_epoch - 1]
        history['eval']['acc'] = history['eval']['acc'][0: start_epoch - 1]

    # 验证集最好的准确率和当时的epoch
    # 为了保持checkpoint的连续性，选取保存最好的准确率和epoch
    if not history['eval']['acc']:
        # 如果没有训练历史记录，设为0
        best_acc = 0
        best_epoch = 0
    else:
        #如果history存在，加载最好的acc和epoch
        best_acc = max(history['eval']['acc'])
        best_epoch = history['eval']['acc'].index(
            max(history['eval']['acc'])) + 1

    # 加载最好的acc和epoch时的模型状态，假设当时也是这样保存的
    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

    #设置device
    device = select_device(gpu)
    print(f'\n开始在device {device} 进行第 {start_epoch} epoch训练')

    #开始训练模型, 起始时间
    train_start = time.time()

    model.to(device)
    #循环epoch
    for epoch in range(start_epoch, start_epoch + epochs):
        print(f'\nEpoch {epoch}/{start_epoch + epochs - 1}:'
              f'\n---------------------')

        # 每个epoch都会进行训练和验证
        for phase in ['train', 'eval']:
            #开始时间
            phase_start = time.time()

            #判断模型时训练吗
            if phase == 'train':
                model.train()
            else:
                model.eval()

            #重置损失和准确度
            running_loss = 0
            running_corrects = 0

            #获取输入的向量和对于的类别标签
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)

                # 当训练时，重置梯度参数
                if phase == 'train':
                    optimizer.zero_grad()
                #是否开启梯度累积
                with torch.set_grad_enabled(phase == 'train'):
                    #前向传播
                    logps = model(inputs)
                    #计算损失
                    loss = criterion(logps, labels)

                    if phase == 'train':
                        #反向传播，只有训练时会反向
                        loss.backward()
                        #更新模型权重参数
                        optimizer.step()

                #更新损失，累加
                running_loss += loss.item() * inputs.size(0)

                # 计算识别正确的个数
                ps = torch.exp(logps)  #可能性
                _, predictions = ps.topk(1, dim=1)   #选取topk，默认为1
                #分类正确的个数,[[True,False,True]]等
                equals = predictions == labels.view(*predictions.shape)
                #分类正确的个数累加, 11.0
                running_corrects += torch.sum(
                    equals.type(torch.FloatTensor)).item()

            #当前阶段的epoch的损失计算
            phase_loss = running_loss / len(dataloaders[phase].dataset)
            history[phase]['loss'].append(phase_loss)

            # 当前阶段的epoch的准确率计算
            phase_acc = running_corrects / len(dataloaders[phase].dataset)
            history[phase]['acc'].append(phase_acc)

            #如果模型的准确率超过历史最好的准确率，保存模型
            if phase == 'eval' and phase_acc > best_acc:
                best_epoch = epoch
                best_acc = phase_acc
                best_model_state_dict = copy.deepcopy(model.state_dict())
                best_optimizer_state_dict = copy.deepcopy(
                    optimizer.state_dict())

            #打印当前阶段结果，耗时和准确率，损失
            phase_duration = time.time() - phase_start
            print(f'{phase.upper()} 完成，耗时 {phase_duration:.0f}s. '
                  f'损失: {phase_loss:.4f}, 准确率: {phase_acc:.4f}')

    # 训练完成epochs后，更新模型状态和优化器，保存最好的状态
    model.load_state_dict(best_model_state_dict)
    optimizer.load_state_dict(best_optimizer_state_dict)

    #打印总体的训练结果
    train_duration = time.time() - train_start
    print(f'\n训练完成，耗时 {(train_duration // 60):.0f}m '
          f'{(train_duration % 60):.0f}s. '
          f'当训练到第 {best_epoch}epoch， '
          f'验证集最好的准确率是: {best_acc:.4f}')

    return history, best_epoch


def plot_history(history):
    """
    绘制训练集和验证集的准确率和损失图表
    Args:
        history (dict):  {'train': {'loss': [], 'acc': []},
                         'eval': {'loss': [], 'acc': []}}
    """
    fig, ax1 = plt.subplots()
    #设置字体
    matplotlib.rcParams['font.family'] = ['Kaiti']
    #更改epochs从1开始
    epochs = np.arange(1, len(history['train']['loss']) + 1)

    #绘制损失
    tl, = ax1.plot(epochs, history['train']['loss'], 'g-', label='Training Loss')
    vl, = ax1.plot(epochs, history['eval']['loss'], 'b-', label='Validation Loss')
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Loss')
    leg1 = ax1.legend(loc='lower right')

    #绘制准确率
    ax2 = ax1.twinx()
    ta, = ax2.plot(epochs, history['train']['acc'], 'y-')
    va, = ax2.plot(epochs, history['eval']['acc'], 'r-')
    ax2.set_ylabel('Accuracy')
    leg2 = ax1.legend([ta, va], ['Training Accuracy','Validation Accuracy'],
                      loc='upper right')
    ax1.add_artist(leg1)
    plt.legend(frameon=False)
    plt.show()


def save_checkpoint(save_path, epoch, model, optimizer, history):
    """
    保存PyTorch的checkpoint通过给定的model，同时保存optimizer和history，epoch
    Args:
        save_path (str): 保存路径，需要携带具体名称
        epoch (int): 保存的epoch
        model: 要保存的模型
        optimizer (torch.optim.Optimizer) 保存的优化器
        history (dict(dict)): 嵌入的字典，包括训练集和验证集的损失和准确率
    """
    print(f'\n正在保存训练效果最好的epoch {epoch} 的checkpoint')

    #如果保存的文件夹不存在，自动创建
    if not os.path.exists(os.path.dirname(save_path)):
        try:
            os.makedirs(os.path.dirname(save_path))
        except OSError as e: #即使存在也不报错
            if e.errno != errno.EEXIST:
                raise

    #更改模型到cpu
    model.to('cpu')

    #设置保存checkpoint的内容
    checkpoint = {
        'arch': model.arch,
        'output_size': model.output_size,
        'class_to_idx': model.class_to_idx,
        'hidden_units': model.hidden_units,
        'drop_p': model.drop_p,
        'history': history,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    #保存模型
    torch.save(checkpoint, save_path)

    #打印保存状态
    file_size = os.path.getsize(save_path)
    print(f'\n模型Checkpoint已保存: {(file_size / 1e6):.2f}Mb\n')


def test_model(dataloader, model, gpu=False):
    """
    测试模型性能，并打印准确率
    Args:
        dataloader (DataLoader)
        model (torchvision model)
        gpu (bool): 是否使用gpu

    Returns:
        test_acc (float):模型准确率
    """
    print('\n开始在测试集上评估模型')
    # 设置模型为评估模式
    model.eval()
    device = select_device(gpu)
    model.to(device)


    #损失函数
    criterion = nn.NLLLoss()

    running_corrects = 0

    with ShadyBar('Progress', max=len(dataloader)) as bar:
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                #前向计算
                logps = model(inputs)
                #计算损失，没用到，测试，不用计算损失
                loss = criterion(logps, labels)

            #计算准确率
            ps = torch.exp(logps)  # probabilities
            _, predictions = ps.topk(1, dim=1)   # top predictions
            equals = predictions == labels.view(*predictions.shape)
            running_corrects += torch.sum(equals.type(torch.FloatTensor)).item()
            bar.next()  #更新进度条

    #计算平均准确率
    test_acc = running_corrects / len(dataloader.dataset)
    return test_acc
