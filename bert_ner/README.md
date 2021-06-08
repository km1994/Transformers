# 【关于 pytorch_pretrained_bert 做命名实体识别】那些你不知道的事

> 作者：杨夕
> 
> 论文学习项目地址：https://github.com/km1994/nlp_paper_study
> 
> 《NLP 百面百搭》地址：https://github.com/km1994/NLP-Interview-Notes
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 

![](img/微信截图_20210301212242.png)

> NLP && 推荐学习群【人数满了，加微信 blqkm601 】

![](output/images/20210523220743.png)

- [【关于 pytorch_pretrained_bert 做命名实体识别】那些你不知道的事](#关于-pytorch_pretrained_bert-做命名实体识别那些你不知道的事)
  - [一、前言](#一前言)
  - [二、实现功能](#二实现功能)
  - [三、所需环境](#三所需环境)
  - [四、项目目录介绍](#四项目目录介绍)
  - [五、训练数据格式](#五训练数据格式)
    - [5.1 前言](#51-前言)
    - [5.2 sent: 以句子形式按行存储](#52-sent-以句子形式按行存储)
    - [5.3 word: 以字形式按行存](#53-word-以字形式按行存)
  - [六、项目使用](#六项目使用)
    - [6.1 模型微调](#61-模型微调)
    - [6.2 模型验证](#62-模型验证)
    - [6.3 模型预测](#63-模型预测)
  - [七、核心代码介绍](#七核心代码介绍)
    - [7.1 前言](#71-前言)
    - [7.2 网络介绍](#72-网络介绍)
      - [7.2.1 Bert_CRF 介绍](#721-bert_crf-介绍)
      - [7.2.2 CRF 介绍](#722-crf-介绍)
    - [7.3 数据预处理模块介绍](#73-数据预处理模块介绍)
      - [7.3.1 输入实例类 定义](#731-输入实例类-定义)
      - [7.3.2 数据预处理的基类定义](#732-数据预处理的基类定义)
      - [7.3.2 自定义的 数据预处理的 MyPro 定义](#732-自定义的-数据预处理的-mypro-定义)
    - [7.4 模型输入特征模块介绍](#74-模型输入特征模块介绍)
      - [7.4.1 模型输入特征类模块介绍](#741-模型输入特征类模块介绍)
      - [7.4.2 examples to 特征](#742-examples-to-特征)
    - [7.5 构造 迭代器 模块介绍](#75-构造-迭代器-模块介绍)
      - [7.5.1 构造 batch 迭代器](#751-构造-batch-迭代器)
    - [7.6 其他模块介绍](#76-其他模块介绍)
  - [参考](#参考)

## 一、前言

本文主要应用 pytorch_pretrained_bert 调用 Bert 模型做中文 命名实体识别 任务。

## 二、实现功能

- 利用 pytorch_pretrained_bert 调用 Bert 模型做中文 命名实体识别；
- 支持 CPU 和 GPU 间切换；
- 支持模型训练、验证、预测等功能；
- 支持多种数据加载；

## 三、所需环境

- python 3.6+
- pytorch 1.6.0+
- pytorch-pretrained-bert  0.6.2
- pandas
- numpy
- sklearn

## 四、项目目录介绍

- Config.py：           配置文件
- main.py：             模型入口
- train.py：            模型训练
- eval.py：             模型验证
- test.py：             模型测试
- inference.py：        模型预测
- data_loader.py：      数据迭代器构建
- data_processor.py：   数据预处理
- data/                 数据存储目录
- net/                  网络
  - crf.py：            CRF
  - bert_ner.py:        Bert-CRF
- util/                 模块目录
  - Logginger.py        日志模块
  - metrics.py          定义性能指标模块
  - model_util.py       模型存储和加载模块
  - plot_util.py        绘制 train 和 eval 的 loss 曲线模块
  - porgress_util.py    进度条 模块
- 

## 五、训练数据格式

### 5.1 前言

本项目 支持 sent（以句子形式按行存储）和 word（以字形式按行存储），两种模式，在选取 对应模式之前，需要在 Config.py 中配置 成员变量 data_type。【注：其他类型数据格式，目前正在增加ing!!!】

```python
    class Config():
        def __init__(self):
            ...
            self.data_type = "word"              # 训练数据类型    sent: 以句子形式按行存储  word: 以字形式按行存储
            if self.data_type == "sent":
                self.data_dir = "data/ccks/"  # 原始数据文件夹，应包括tsv文件
                self.train_data_file = f"{self.data_dir}train_sent.txt"
                self.valid_data_file = f"{self.data_dir}valid_sent.txt"
                self.labels = ["S-seg", "B-seg", "I-seg", "E-seg", "O"]
                self.lable_type = "BIESO"
                self.tag_map = {label:i for i,label in enumerate(self.labels)}
            else:
                self.data_dir = "data/msra/"  # 原始数据文件夹，应包括tsv文件
                self.train_data_file = f"{self.data_dir}example.train"
                self.valid_data_file = f"{self.data_dir}example.dev"
                self.labels = ["B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER", "O"]
                self.lable_type = "BIO"
                self.tag_map = {label:i for i,label in enumerate(self.labels)}
            ...
```

### 5.2 sent: 以句子形式按行存储  

当 Config.py 中配置 成员变量 data_type 为 sent（以句子形式按行存储），对应数据格式为：

```s
    莫 妮 卡 · 贝 鲁 奇 的 代 表 作 ？	B-seg I-seg I-seg I-seg I-seg I-seg E-seg O O O O O
    《 湖 上 草 》 是 谁 的 诗 ？	O B-seg I-seg E-seg O O O O O O
    龙 卷 风 的 英 文 名 是 什 么 ？	B-seg I-seg E-seg O O O O O O O O
```

### 5.3 word: 以字形式按行存

当 Config.py 中配置 成员变量 data_type 为 word（以字形式按行存储），对应数据格式为：

```s
    海 O
    钓 O
    比 O
    赛 O
    地 O
    点 O
    在 O
    厦 B-LOC
    门 I-LOC
    与 O
    金 B-LOC
    门 I-LOC
    之 O
    间 O
    的 O
    海 O
    域 O
    。 O

    ...
```

## 六、项目使用

### 6.1 模型微调

```python
    $python main.py
    >>>
    ...
    [2021-06-08 17:02:11]: bert_ner data_processor.py[line:406] INFO  tokens: [CLS] 在 这 里 恕 弟 不 恭 之 罪 ， 敢 在 尊 前 一 诤 ： 前 人 论 书 ， 每 曰 “ 字 字 有 来 历 ， 笔 笔 有 出 处 ” ， 细 读 公 字 ， 何 尝 跳 出 前 人 藩 篱 ， 自 隶 变 而 后 ， 直 至 明 季 ， 兄 有 何 新 出 ？ [SEP]
    [2021-06-08 17:02:11]: bert_ner data_processor.py[line:407] INFO  input_ids: 101 1762 6821 7027 2609 2475 679 2621 722 5389 8024 3140 1762 2203 1184 671 6420 8038 1184 782 6389 741 8024 3680 3288 100 2099 2099 3300 3341 1325 8024 5011 5011 3300 1139 1905 100 8024 5301 6438 1062 2099 8024 862 2214 6663 1139 1184 782 5974 5075 8024 5632 7405 1359 5445 1400 8024 4684 5635 3209 2108 8024 1040 3300 862 3173 1139 8043 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    [2021-06-08 17:02:11]: bert_ner data_processor.py[line:408] INFO  input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    [2021-06-08 17:02:11]: bert_ner data_processor.py[line:409] INFO  label: 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 
    [2021-06-08 17:02:11]: bert_ner data_processor.py[line:410] INFO  output_mask: 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    [2021-06-08 17:02:11]: bert_ner data_loader.py[line:91] INFO    Num examples = 2318
    [2021-06-08 17:02:11]: bert_ner data_loader.py[line:92] INFO    Batch size = 256
    [2021-06-08 17:02:11]: bert_ner main.py[line:23] INFO  fit
    [2021-06-08 17:02:11]: bert_ner train.py[line:87] INFO  start train!!!
    nohup: ignoring input
    [2021-06-08 17:03:03]: bert_ner data_loader.py[line:76] INFO    Num steps = 1630
    [2021-06-08 17:03:03]: bert_ner data_processor.py[line:404] INFO  -----------------Example-----------------
    [2021-06-08 17:03:03]: bert_ner data_processor.py[line:405] INFO  guid: train - 0
    [2021-06-08 17:03:03]: bert_ner data_processor.py[line:406] INFO  tokens: [CLS] 海 钓 比 赛 地 点 在 厦 门 与 金 门 之 间 的 海 域 。 [SEP]
    [2021-06-08 17:03:03]: bert_ner data_processor.py[line:407] INFO  input_ids: 101 3862 7157 3683 6612 1765 4157 1762 1336 7305 680 7032 7305 722 7313 4638 3862 1818 511 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    [2021-06-08 17:03:03]: bert_ner data_processor.py[line:408] INFO  input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    [2021-06-08 17:03:03]: bert_ner data_processor.py[line:409] INFO  label: 4 4 4 4 4 4 4 5 6 4 5 6 4 4 4 4 4 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 
    [2021-06-08 17:03:03]: bert_ner data_processor.py[line:410] INFO  output_mask: 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    [2021-06-08 17:03:07]: bert_ner data_loader.py[line:91] INFO    Num examples = 20864
    [2021-06-08 17:03:07]: bert_ner data_loader.py[line:92] INFO    Batch size = 256
    [2021-06-08 17:03:08]: bert_ner data_processor.py[line:404] INFO  -----------------Example-----------------
    [2021-06-08 17:03:08]: bert_ner data_processor.py[line:405] INFO  guid: valid - 0
    [2021-06-08 17:03:08]: bert_ner data_processor.py[line:406] INFO  tokens: [CLS] 在 这 里 恕 弟 不 恭 之 罪 ， 敢 在 尊 前 一 诤 ： 前 人 论 书 ， 每 曰 “ 字 字 有 来 历 ， 笔 笔 有 出 处 ” ， 细 读 公 字 ， 何 尝 跳 出 前 人 藩 篱 ， 自 隶 变 而 后 ， 直 至 明 季 ， 兄 有 何 新 出 ？ [SEP]
    [2021-06-08 17:03:08]: bert_ner data_processor.py[line:407] INFO  input_ids: 101 1762 6821 7027 2609 2475 679 2621 722 5389 8024 3140 1762 2203 1184 671 6420 8038 1184 782 6389 741 8024 3680 3288 100 2099 2099 3300 3341 1325 8024 5011 5011 3300 1139 1905 100 8024 5301 6438 1062 2099 8024 862 2214 6663 1139 1184 782 5974 5075 8024 5632 7405 1359 5445 1400 8024 4684 5635 3209 2108 8024 1040 3300 862 3173 1139 8043 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    [2021-06-08 17:03:08]: bert_ner data_processor.py[line:408] INFO  input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    [2021-06-08 17:03:08]: bert_ner data_processor.py[line:409] INFO  label: 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 
    [2021-06-08 17:03:08]: bert_ner data_processor.py[line:410] INFO  output_mask: 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    [2021-06-08 17:03:08]: bert_ner data_loader.py[line:91] INFO    Num examples = 2318
    [2021-06-08 17:03:08]: bert_ner data_loader.py[line:92] INFO    Batch size = 256
    [2021-06-08 17:03:08]: bert_ner main.py[line:23] INFO  fit
    [2021-06-08 17:03:08]: bert_ner train.py[line:87] INFO  start train!!!
    [2021-06-08 17:04:19]: bert_ner train.py[line:125] INFO  step:0=>loss:2.0037906169891357,train_acc:0.10507805798593241
    [2021-06-08 17:05:53]: bert_ner train.py[line:125] INFO  step:0=>loss:1.8787178993225098,train_acc:0.25330245325098644
    ...
    Epoch 4 - train_loss: 0.039489 - eval_loss: 0.059254 - train_acc:0.975821 - eval_acc:0.974665 - p:0.944604 - r:0.949216 - f1:0.946441 - best f1:0.946441
    ...
```

### 6.2 模型验证

```python   
    $ python eval.py
    >>>

```

### 6.3 模型预测

```python
    $python inference.py 
    >>>
    输入query：海钓比赛地点在厦门与金门之间的海域。

    query:海钓比赛地点在厦门与金门之间的海域。
    paths:[3, 6, 6, 6, 6, 6, 6, 2, 3, 6, 2, 3, 6, 6, 6, 6, 6, 6]
    pre_tags:['I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O']
    entity:[('厦门', 'ORG', 7, 8), ('金门', 'ORG', 10, 11)]
    ...
```

## 七、核心代码介绍

### 7.1 前言

该代码的组织架构 和 Bert tensorflow 版本差不多。

### 7.2 网络介绍

#### 7.2.1 Bert_CRF 介绍

```python
import torch.nn as nn
from net.crf import CRF
import numpy as np
from sklearn.metrics import f1_score, classification_report
from pytorch_pretrained_bert.modeling import BertForTokenClassification
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from Config import Config
args = Config()

class Bert_CRF(nn.Module):
    def __init__(self):
        super(Bert_CRF, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(
            args.bert_model, 
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
            num_labels=len(args.labels)
        )
        self.crf = CRF(len(args.labels))

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        label_id=None,
        output_all_encoded_layers=False
    ):
        logit = self.bert(input_ids, token_type_ids)
        return logit
    
    # 功能：计算 loss 
    def loss_fn(self, bert_encode, output_mask, tags):
        loss = self.crf.negative_log_loss(bert_encode, output_mask, tags)
        return loss

    # 功能：预测
    def predict(self, bert_encode, output_mask):
        predicts = self.crf.get_batch_best_path(bert_encode, output_mask)
        # predicts = predicts.view(1, -1).squeeze()
        # predicts = predicts[predicts != -1]
        return predicts

    # 功能：计算 acc 和 f1
    def acc_f1(self, y_pred, y_true):
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        f1 = f1_score(y_true, y_pred, average="macro")
        correct = np.sum((y_true == y_pred).astype(int))
        acc = correct / y_pred.shape[0]
        return acc, f1

    # 功能：计算 评价指标
    def class_report(self, y_pred, y_true):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)
```

#### 7.2.2 CRF 介绍

```python
import torch
import torch.nn as nn
from torch.autograd import Variable

class CRF(nn.Module):
    """线性条件随机场"""
    def __init__(self, num_tag, use_cuda=False):
        if num_tag <= 0:
            raise ValueError("Invalid value of num_tag: %d" % num_tag)
        super(CRF, self).__init__()
        self.num_tag = num_tag
        self.start_tag = num_tag
        self.end_tag = num_tag + 1
        self.use_cuda = use_cuda
        # 转移矩阵transitions：P_jk 表示从tag_j到tag_k的概率
        # P_j* 表示所有从tag_j出发的边
        # P_*k 表示所有到tag_k的边
        self.transitions = nn.Parameter(torch.Tensor(num_tag + 2, num_tag + 2))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[self.end_tag, :] = -10000  # 表示从EOS->其他标签为不可能事件, 如果发生，则产生一个极大的损失
        self.transitions.data[:, self.start_tag] = -10000  # 表示从其他标签->SOS为不可能事件, 同上

    def real_path_score(self, features, tags):
        """
        features: (time_steps, num_tag)
        real_path_score表示真实路径分数
        它由Emission score和Transition score两部分相加组成
        Emission score由LSTM输出结合真实的tag决定，表示我们希望由输出得到真实的标签
        Transition score则是crf层需要进行训练的参数，它是随机初始化的，表示标签序列前后间的约束关系（转移概率）
        Transition矩阵存储的是标签序列相互间的约束关系
        在训练的过程中，希望real_path_score最高，因为这是所有路径中最可能的路径
        """
        r = torch.LongTensor(range(features.size(0)))
        if self.use_cuda:
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.start_tag]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.end_tag])])
            r = r.cuda()
        else:
            pad_start_tags = torch.cat([torch.LongTensor([self.start_tag]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.end_tag])])
        # Transition score + Emission score
        score = torch.sum(self.transitions[pad_start_tags, pad_stop_tags]).cpu() + torch.sum(features[r, tags])
        return score

    def all_possible_path_score(self, features):
        """
        计算所有可能的路径分数的log和：前向算法
        step1: 将forward列expand成3*3
        step2: 将下个单词的emission行expand成3*3
        step3: 将1和2和对应位置的转移矩阵相加
        step4: 更新forward，合并行
        step5: 取forward指数的对数计算total
        """
        time_steps = features.size(0)
        # 初始化
        forward = Variable(torch.zeros(self.num_tag))  # 初始化START_TAG的发射分数为0
        if self.use_cuda:
            forward = forward.cuda()
        for i in range(0, time_steps):  # START_TAG -> 1st word -> 2nd word ->...->END_TAG
            emission_start = forward.expand(self.num_tag, self.num_tag).t()
            emission_end = features[i, :].expand(self.num_tag, self.num_tag)
            if i == 0:
                trans_score = self.transitions[self.start_tag, :self.start_tag].cpu()
            else:
                trans_score = self.transitions[:self.start_tag, :self.start_tag].cpu()
            sum = emission_start + emission_end + trans_score
            forward = log_sum(sum, dim=0)
        forward = forward + self.transitions[:self.start_tag, self.end_tag].cpu()  # END_TAG
        total_score = log_sum(forward, dim=0)
        return total_score

    def negative_log_loss(self, inputs, output_mask, tags):
        """
        inputs:(batch_size, time_step, num_tag)
        target_function = P_real_path_score/P_all_possible_path_score
                        = exp(S_real_path_score)/ sum(exp(certain_path_score))
        我们希望P_real_path_score的概率越高越好，即target_function的值越大越好
        因此，loss_function取其相反数，越小越好
        loss_function = -log(target_function)
                      = -S_real_path_score + log(exp(S_1 + exp(S_2) + exp(S_3) + ...))
                      = -S_real_path_score + log(all_possible_path_score)
        """
        if not self.use_cuda:
            inputs = inputs.cpu()
            output_mask = output_mask.cpu()
            tags = tags.cpu()

        loss = Variable(torch.tensor(0.), requires_grad=True)
        num_tag = inputs.size(2)
        num_chars = torch.sum(output_mask.detach()).float()
        for ix, (features, tag) in enumerate(zip(inputs, tags)):
            # 过滤[CLS] [SEP] sub_word
            # features (time_steps, num_tag)
            # output_mask (batch_size, time_step)
            num_valid = torch.sum(output_mask[ix].detach())
            features = features[output_mask[ix] == 1]
            tag = tag[:num_valid]
            real_score = self.real_path_score(features, tag)
            total_score = self.all_possible_path_score(features)
            cost = total_score - real_score
            loss = loss + cost
        return loss / num_chars

    def viterbi(self, features):
        time_steps = features.size(0)
        forward = Variable(torch.zeros(self.num_tag))  # START_TAG
        if self.use_cuda:
            forward = forward.cuda()
        # back_points 到该点的最大分数  last_points 前一个点的索引
        back_points, index_points = [self.transitions[self.start_tag, :self.start_tag].cpu()], [
            torch.LongTensor([-1]).expand_as(forward)]
        for i in range(1, time_steps):  # START_TAG -> 1st word -> 2nd word ->...->END_TAG
            emission_start = forward.expand(self.num_tag, self.num_tag).t()
            emission_end = features[i, :].expand(self.num_tag, self.num_tag)
            trans_score = self.transitions[:self.start_tag, :self.start_tag].cpu()
            sum = emission_start + emission_end + trans_score
            forward, index = torch.max(sum.detach(), dim=0)
            back_points.append(forward)
            index_points.append(index)
        back_points.append(forward + self.transitions[:self.start_tag, self.end_tag].cpu())  # END_TAG
        return back_points, index_points

    def get_best_path(self, features):
        back_points, index_points = self.viterbi(features)
        # 找到线头
        best_last_point = argmax(back_points[-1])
        index_points = torch.stack(index_points)  # 堆成矩阵
        m = index_points.size(0)
        # 初始化矩阵
        best_path = [best_last_point]
        # 循着线头找到其对应的最佳路径
        for i in range(m - 1, 0, -1):
            best_index_point = index_points[i][best_last_point]
            best_path.append(best_index_point)
            best_last_point = best_index_point
        best_path.reverse()
        return best_path

    def get_batch_best_path(self, inputs, output_mask):
        if not self.use_cuda:
            inputs = inputs.cpu()
            output_mask = output_mask.cpu()
        batch_best_path = []
        max_len = inputs.size(1)
        num_tag = inputs.size(2)
        for ix, features in enumerate(inputs):
            features = features[output_mask[ix] == 1]
            best_path = self.get_best_path(features)
            best_path = torch.Tensor(best_path).long()
            best_path = padding(best_path, max_len)
            batch_best_path.append(best_path)
        batch_best_path = torch.stack(batch_best_path, dim=0)
        return batch_best_path

def log_sum(matrix, dim):
    """
    前向算法是不断累积之前的结果，这样就会有个缺点
    指数和累积到一定程度后，会超过计算机浮点值的最大值，变成inf，这样取log后也是inf
    为了避免这种情况，我们做了改动：
    1. 用一个合适的值clip去提指数和的公因子，这样就不会使某项变得过大而无法计算
    SUM = log(exp(s1)+exp(s2)+...+exp(s100))
        = log{exp(clip)*[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]}
        = clip + log[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]
    where clip=max
    """
    clip_value = torch.max(matrix)  # 极大值
    clip_value = int(clip_value.data.tolist())
    log_sum_value = clip_value + torch.log(torch.sum(torch.exp(matrix - clip_value), dim=dim))
    return log_sum_value

def argmax(matrix, dim=0):
    """(0.5, 0.4, 0.3)"""
    _, index = torch.max(matrix, dim=dim)
    return index

def padding(vec, max_len, pad_token=-1):
    new_vec = torch.zeros(max_len).long()
    new_vec[:vec.size(0)] = vec
    new_vec[vec.size(0):] = pad_token
    return new_vec

```

### 7.3 数据预处理模块介绍

#### 7.3.1 输入实例类 定义

```python
# 功能：输入实例
class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """
            功能：创建一个输入实例
            input:
                guid: 每个example拥有唯一的id
                text_a: 第一个句子的原始文本，一般对于文本分类来说，只需要text_a
                text_b: 第二个句子的原始文本，在句子对的任务中才有，分类问题中为None
                label: example对应的标签，对于训练集和验证集应非None，测试集为None
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
```

#### 7.3.2 数据预处理的基类定义

```python
# 功能：数据预处理的基类，自定义的MyPro继承该类
class DataProcessor(object):
    """数据预处理的基类，自定义的MyPro继承该类"""

    def get_valid_examples(self, data_dir):
        """读取验证集 Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """读取验证集 Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """读取标签 Gets the list of labels for this data set."""
        raise NotImplementedError()
```

#### 7.3.2 自定义的 数据预处理的 MyPro 定义

```python
# 功能：将数据构造成example格式
class MyPro(DataProcessor):
    """
        功能：将数据构造成example格式
    """
    def __init__(self):
        self.make_example_list_fun_dic = {
            "sent":self.sent_make_example_list,
            "word":self.word_make_example_list
        }

    # 功能：获取验证集
    def get_valid_examples(self):
        valid_examples = self.make_example_list_fun_dic[args.data_type]('valid',args.valid_data_file) 
        return valid_examples
    # 功能：获取验证集
    def get_train_examples(self):
        train_examples = self.make_example_list_fun_dic[args.data_type]('train',args.train_data_file)
        return train_examples
    # 功能：获取标签集
    def get_labels(self):
        return args.labels

    # 功能：将数据转化为 实例
    def sent_make_example_list(self, stage, file_name):
        '''
            功能：将数据转化为 实例
            input:
                stage           String              模式  train or valid
                file_name       String              文件名称
            return:
                examples        List(InputExample)  数据实例
        '''
        i = 0
        examples = []
        labels_set = set()
        with open(file_name,encoding="utf-8",mode="r") as fr:
            lines = fr.readlines()
            for line in lines:
                seq,label = line.strip().split("\t")
                label = label.split(args.sep)
                guid = "%s - %d" % (stage, i)
                example = InputExample(
                    guid=guid, 
                    text_a=seq.split(args.sep), 
                    label=label
                )
                examples.append(example)
                labels_set = labels_set.union(label)
        if stage=="train":
            if stage=="train":
                args.labels = list(labels_set)
            else:
                assert labels_set.issubset(args.labels), f"valid labels {labels_set} is not train labels {args.labels} subset!"
        return examples

     # 功能：将数据转化为 实例
    def word_make_example_list(self, stage, file_name):
        '''
            功能：将数据转化为 实例
            input:
                stage           String              模式  train or valid
                file_name       String              文件名称
            return:
                examples        List(InputExample)  数据实例
        '''
        i = 0
        examples = []
        labels_set = set()
        with open(file_name,encoding="utf-8",mode="r") as fr:
            lines = fr.readlines()
            with open(file_name,encoding="utf-8",mode="r") as fr:
                lines = fr.readlines()
                sent = []
                tags = []
                for line in lines:
                    if line != "\n":
                        word,tag = line.strip().split(args.sep)
                        sent.append(word)
                        tags.append(tag)
                    else:
                        guid = "%s - %d" % (stage, i)
                        example = InputExample(
                            guid=guid, 
                            text_a=sent, 
                            label=tags
                        )
                        examples.append(example)
                        labels_set = labels_set.union(tags)
                        sent = []
                        tags = []
        if stage=="train":
            if stage=="train":
                args.labels = list(labels_set)
            else:
                assert labels_set.issubset(args.labels), f"valid labels {labels_set} is not train labels {args.labels} subset!"
        return examples
```

### 7.4 模型输入特征模块介绍

#### 7.4.1 模型输入特征类模块介绍

```python
# 功能：模型输入特征
class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, output_mask):
        """
            功能：模型输入特征
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.output_mask = output_mask
```

#### 7.4.2 examples to 特征

```python
# 功能：词转换成数字
def convert_text_to_ids(text, tokenizer):
    '''
        功能：词转换成数字
        input:
            text           String               句子
            tokenizer       Bert tokenizer
        return:
            ids             List                句子 id
    '''
    ids = []
    for t in text:
        if t not in tokenizer.vocab:
            t = '[UNK]'
        ids.append(tokenizer.vocab.get(t))
    return ids

# 功能：examples to 特征
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    '''
        功能：examples to 特征
        input:
            examples        examples        实例
            label_list      List            标签 List
            max_seq_length  int             max_len
            tokenizer       BertTokenizer   
        return:
            features        List            模型 输入 features
    '''
    # 标签转换为数字
    label_map = {label: i for i, label in enumerate(label_list)}

    # load sub_vocab
    sub_vocab = {}
    with open(args.VOCAB_FILE, 'r',encoding="utf-8") as fr:
        for line in fr:
            _line = line.strip('\n')
            if "##" in _line and sub_vocab.get(_line) is None:
                sub_vocab[_line] = 1

    features = []
    for ex_index, example in enumerate(examples):
        tokens_a = example.text_a
        labels = example.label

        if len(tokens_a) == 0 or len(labels) == 0:
            continue

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
            labels = labels[:(max_seq_length - 2)]

        # ----------------处理 source--------------
        ## 句子首尾加入标示符
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        ## 词转换成数字
        input_ids = convert_text_to_ids(tokens, tokenizer)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # ---------------处理 target----------------
        ## Notes: label_id 中不包括[CLS]和[SEP]
        label_id = [label_map[l] for l in labels]
        label_padding = [-1] * (max_seq_length - len(label_id))
        label_id += label_padding

        ## output_mask 用来过滤 bert 输出中 sub_word 的输出,只保留单词的第一个输出(As recommended by jocob in his paper)
        ## 此外，也是为了适应crf
        output_mask = [1 for t in tokens_a]
        output_mask = [0] + output_mask + [0]
        output_mask += padding

        ot = torch.LongTensor(output_mask)
        lt = torch.LongTensor(label_id)

        if len(ot[ot == 1]) != len(lt[lt != -1]):
            print(tokens_a)
            print(ot[ot == 1])
            print(lt[lt != -1])
            print(len(ot[ot == 1]), len(lt[lt != -1]))
        assert len(ot[ot == 1]) == len(lt[lt != -1])

        # ----------------处理后结果-------------------------
        # for example, in the case of max_seq_length=10:
        # raw_data:          春 秋 忽 代 谢le
        # token:       [CLS] 春 秋 忽 代 谢 ##le [SEP]
        # input_ids:     101 2  12 13 16 14 15   102   0 0 0
        # input_mask:      1 1  1  1  1  1   1     1   0 0 0
        # label_id:          T  T  O  O  O
        # output_mask:     0 1  1  1  1  1   0     0   0 0 0
        # --------------看结果是否合理------------------------

        if ex_index < 1:
            logger.info("-----------------Example-----------------")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("label: %s " % " ".join([str(x) for x in label_id]))
            logger.info("output_mask: %s " % " ".join([str(x) for x in output_mask]))
        # ----------------------------------------------------

        feature = InputFeature(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            output_mask=output_mask
        )
        features.append(feature)
    return features
```

### 7.5 构造 迭代器 模块介绍

#### 7.5.1 构造 batch 迭代器

```python
# 功能：构造 batch 迭代器
def create_batch_iter(mode):
    """
        功能：构造 batch 迭代器
        input:
            mode        String          模式  train or valid
    """
    processor, tokenizer = init_params()
    if mode == "train":
        examples = processor.get_train_examples()

        num_train_steps = int(
            len(examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs
        )

        batch_size = args.train_batch_size

        logger.info("  Num steps = %d", num_train_steps)

    elif mode == "valid":
        examples = processor.get_valid_examples()
        batch_size = args.eval_batch_size
    else:
        raise ValueError("Invalid mode %s" % mode)

    label_list = processor.get_labels()

    # examples to 特征
    features = convert_examples_to_features(
        examples, label_list, args.max_seq_length, tokenizer
    )

    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_output_mask = torch.tensor([f.output_mask for f in features], dtype=torch.long)

    # 数据集
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_output_mask)

    if mode == "train":
        sampler = RandomSampler(data)
    elif mode == "valid":
        sampler = SequentialSampler(data)
    else:
        raise ValueError("Invalid mode %s" % mode)

    # 迭代器
    iterator = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if mode == "train":
        return iterator, num_train_steps
    elif mode == "valid":
        return iterator
    else:
        raise ValueError("Invalid mode %s" % mode)
```

### 7.6 其他模块介绍

其他模块和 基本的 torch 模型构建 差不多，此处不做介绍。

## 参考


