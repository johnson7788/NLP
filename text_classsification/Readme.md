1. data：文件夹中存储的主要是数据
2. nets: package,主要存储网络结构模型代码
    text_cnn: TextCNN做文本分类
3. utils: package,主要存储工具函数相关的代码
    data_helpers.py: 数据加载、批次构建相关函数代码
    network_utils.py: 优化器参数构建相关代码
    vocabulary_utils.py: 词汇转换相关代码
4. train.py: 模型训练入口函数
5. eval.py: 模型效果评估的入口函数
6. graph：模型执行可视化文件保存的文件夹
7. model：模型持久化保存的文件夹
8. deploy：模型部署相关package