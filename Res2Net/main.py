import argparse
from data_utils import load_data, display_prediction, process_image
from config import Config
import os
from model_utils import (
        create_model,  #根据架构创建模型
        create_optimizer,   # 为模型的最后的Classifier层添加优化器
        load_checkpoint,  # 加载checkpoint，重建预训练模型
        plot_history,  # 绘制历史的训练损失和准确率的图表
        save_checkpoint,   #保存模型checkpoint
        train_model,   #训练模型
        classify_image,  #用于预测阶段，使用模型进行预测
        test_model)  #使用测试集测试模型性能，并打印准确率

def train():
        conf = Config()
        # 打印模型配置信息
        conf.dump()
        parser = argparse.ArgumentParser(description='图片分类模型训练')
        parser.add_argument(
                '--resume_checkpoint', action='store', type=str, default='model/checkpoint.pth',
                help='从模型的checkpoint恢复模型，并继续训练，如果resume_checkpoint这个参数提供'
                     '这些参数将忽略--arch, --learning_rate, --hidden_units, and --drop_p')
        args = parser.parse_args()

        #加载数据
        dataloaders, class_to_idx = load_data(conf.data_directory)

        #创建模型，如果模型文件存在
        if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
                #加载checkpoint
                print('resume_checkpoint已存在，开始加载模型')
                model, optimizer, epoch, history = load_checkpoint(
                        checkpoint_path=args.resume_checkpoint,
                        load_optimizer=True, gpu=conf.cuda)
                start_epoch = epoch + 1
        else:
                #创建新模型和优化器
                print('resume_checkpoint未设置或模型文件不存在，创建新的模型')
                model = create_model(
                        arch=conf.arch, class_to_idx=class_to_idx,
                        hidden_units=conf.hidden_units, drop_p=conf.dropout)
                optimizer = create_optimizer(model=model, lr=conf.learning_rate)
                start_epoch = 1
                history = None

        #训练模型
        history, best_epoch = train_model(
                dataloaders=dataloaders, model=model,
                optimizer=optimizer, gpu=conf.cuda, start_epoch=start_epoch,
                epochs=conf.epochs, train_history=history)

        #测试集上测试模型
        test_acc = test_model(dataloader=dataloaders['test'], model=model, gpu=conf.cuda)
        print(f'模型在测试集上的准确率是 {(test_acc * 100):.2f}%')

        #保存模型
        save_checkpoint(
                save_path=conf.save_path+conf.save_name, epoch=best_epoch, model=model,
                optimizer=optimizer, history=history)

        #绘制历史记录
        plot_history(history)

def predict():
        conf = Config()
        # 打印模型配置信息
        conf.dump()
        parser = argparse.ArgumentParser(description='图片分类模型训练')
        parser.add_argument(
                '--image_path', type=str,default='data/zhengjian/predict/test/3601216003722.jpg', help='指定要分类的路径')
        parser.add_argument(
                '--checkpoint', type=str, default='model/checkpoint.pth', help='指定checkpoint的模型的保存位置')
        parser.add_argument(
                '--top_k', type=int, default=2, help='选取topk概率的最大类别， dafault=2')
        args = parser.parse_args()

        # 加载转换，处理，转换图片到Tensor
        image_tensor = process_image(image_path=args.image_path)

        # 加载模型，是否使用gpu
        model, _, _, _ = load_checkpoint(
                checkpoint_path=args.checkpoint, load_optimizer=False, gpu=conf.cuda)

        #图片分类
        probabilities, predictions = classify_image(
                image_tensor=image_tensor, model=model, top_k=args.top_k, gpu=conf.cuda)

        #分类结果
        top_class = predictions[0]
        top_prob = probabilities[0]
        top_k = args.top_k
        print(f'\n预测概率最高的类别是 {top_class.capitalize()} '
              f' 概率是{top_prob:.4f}')
        print(f'\n预测的topk是 {top_k} 类别是 {predictions}'
              f'概率是 {probabilities}')

        # 绘图
        display_prediction(
                image_path=args.image_path,
                probabilities=probabilities,
                predictions=predictions)

if __name__ == '__main__':
    train()
    # predict()