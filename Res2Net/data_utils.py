import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
import matplotlib
import pickle

import torch
from torchvision import datasets, transforms


def load_data(path):
    """
    加载，转换，创建torch.utils.data.Dataloaders
    Args:
        path (str): 路径需要包含子文件夹，例如
                    path/train/..
                    path/eval/..
                    path/test/..

    Returns:
        dataloaders (dict): {'train': Dataloader(train_data),
                             'eval':, Dataloader(valid_data),
                             'test': Dataloader(test_data)}
    """
    #训练图片的数据属性
    IMG_SIZE = 224  #训练数据尺寸
    IMG_MEAN = [0.485, 0.456, 0.406]  # 图片归一化均值
    IMG_SDEV = [0.229, 0.224, 0.225]  # 图片归一化标准差

    #训练阶段
    phases = ['train', 'eval', 'test']

    #文件夹路径
    data_dir = {n: path + n for n in phases}

    #设置transforms
    data_transforms = {
        'train':
            transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMG_MEAN, IMG_SDEV)]),
        'eval':
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(IMG_MEAN, IMG_SDEV)]),
        'test':
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(IMG_MEAN, IMG_SDEV)])
    }

    #加载文件生成datasets
    image_datasets = {n: datasets.ImageFolder(
                            data_dir[n], transform=data_transforms[n])
                      for n in phases}

    #创建dataloaders
    dataloaders = {n: torch.utils.data.DataLoader(
                        image_datasets[n], batch_size=64, shuffle=True)
                    for n in phases}

    # 类别到id
    class_to_idx = image_datasets['train'].class_to_idx

    return dataloaders, class_to_idx


def display_prediction(image_path, probabilities, predictions):
    """
    绘制分类图像，将top预测类别作为标题，并显示预测top类别的预测概率图
    Args:
        image_path (str): 分类图片的路径
        probabilities ([float]): topk预测概率的列表
        class_idxs ([int]): topk类别id的列表
        class_names ([str]): topk的类别名称
    """
    top_class = predictions[0]
    #设置字体
    matplotlib.rcParams['font.family'] = ['Kaiti']

    #设置网格和标题
    fig = plt.figure(figsize=(4, 5.4))
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    fig.suptitle(top_class.capitalize(), x=0.6, y=1, fontsize=16)

    #显示图片
    ax1.imshow(Image.open(image_path))
    ax1.set_xticks([])
    ax1.set_yticks([])

    # 显示预测的类别和概率
    #设置y轴
    y = np.arange(len(predictions))
    ax2.barh(y, probabilities)
    ax2.set_yticks(y)
    ax2.set_yticklabels(predictions)
    #预测的最高概率
    ax2.invert_yaxis()
    ax2.set_xlabel('Prediction probability')

    #调整layout
    fig.tight_layout()
    plt.subplots_adjust(top=0.93)

    plt.show()

def prediction_class_names(predictions, class_to_idx):
    """
    转换索引到类别名称
    Args:
        predictions ([int]): 要预测的类别索引
        class_to_idx (dict): 类别到id映射

    Returns:
        class_names ([str]): 返回预测的类别名称
    """
    class_dict = {val: key for key, val in class_to_idx.items()}
    class_idxs = [class_dict[pred] for pred in predictions]

    return class_idxs

def process_image(image_path):
    """
    缩放，裁剪，归一化PIL 图片， 返回一个Numpy数组
    Args:
        image_path : 输入PIL图片的路径

    Returns:
        image_tensor (Tensor): 处理图片，返回torch.FloatTensor
    """
    IMG_SIZE = 224
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_SDEV = [0.229, 0.224, 0.225]

    #加载图片
    image = Image.open(image_path)

    # Resize最大维度256
    if image.size[0] >= image.size[1]:
        image.thumbnail((256, image.size[1] * 256 // image.size[0]))
    else:
        image.thumbnail((image.size[0] * 256 // image.size[1], 256))

    #中间裁切
    image = image.crop((
            (image.size[0] - IMG_SIZE) // 2,
            (image.size[1] - IMG_SIZE) // 2,
            (image.size[0] + IMG_SIZE) // 2 ,
            (image.size[1] + IMG_SIZE) // 2))
    # 转换到np.array ，rescape channels到0-1之间
    image = np.array(image) / 255
    # 归一化图片
    image = (image - np.array(IMG_MEAN)) / np.array(IMG_SDEV)
    # 调整颜色通道到维度1
    image = image.transpose(2, 0, 1)
    # 转换成toch.FloatTensor
    image_tensor = torch.from_numpy(
            np.expand_dims(image, axis=0)).type(torch.FloatTensor)

    return image_tensor


def save_label(label, filename):
    """
    使用pickle保存字典
    :param vocab:
    :param filename:
    :return:
    """
    with open(filename, 'wb') as f:
        pickle.dump(label, f)

def load_label(filename):
    """
    使用pickle加载字典
    :param filename:
    :return:
    """
    with open(filename, 'rb') as f:
        vocab = pickle.load(f)
    return vocab