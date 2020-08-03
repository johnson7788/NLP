### 主要目录结构
```
├── Readme.md
├── config.py    #模型配置文件
├── data      #数据集，包括训练数据和验证数据, 每个子文件是一个类别，里面放对于文本
├── main.py    #模型运行入口，里面主要包含3个函数，分别是训练，测试，和实际运行的预测接口predict
├── model     #保存TextCNN模型和生成的字典
├── model_utils.py   #TextCNN模型文件
└── data_utils.py    #文本预处理模块
```

