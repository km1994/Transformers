# 【HugBert02】热身运动：安装及向量编码

> 作者：杨夕
> 
> github:https://github.com/km1994/
> 
> 参考：https://zhuanlan.zhihu.com/p/143161582

## 1. Transformers库 所需环境准备

Transformers 库 需要依赖以下环节：

- Python 3.6+
- PyTorch 1.0.0+
- TensorFlow 2.0+

代码如下：

```python
    conda create -n transformers pip python=3.7
    conda activate transformers
    # linux with cuda 10.2
    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 
    conda install tensorflow=2
    # some more useful packages to install, optional
    conda install ipython pandas
```

## 2. jupyter notebook 添加 conda 环境

- 问题：在conda原有环境的基础上,添加了新的conda环境，需要在新环境下使用jupyter notebook
- 解决方法：

1. 安装ipykernel

```
    conda install ipykernel  
```

2. 激活conda环境

```
    source activate transformers
```

3. 将环境写入notebook的kernel中

```
    python -m ipykernel install --user --name transformers --display-name transformers
```

4. 打开notebook

```
    jupyter notebook  
```


## 3. Transformers 库 安装

Transformers 库 有两种 安装 方式：

- 第一种：通过 pip 安装

```
    pip install transformers
```

- 第二种：直接从源代码安装，如果后续要完整运行transformers自带的最新examples的话，建议用这种方式安装

```python
    git clone https://github.com/huggingface/transformers
    cd transformers
    pip install .
```

后续代码更新升级：

```
git pull
pip install --upgrade .
```

最后，检查一下安装的正确性。

```python
    import tensorflow as tf; tf.__version__ # 2.0.0
    import torch; torch.__version__ # 1.3.1
    import transformers;transformers.__version__ # 2.10.0
```

## 4. 加载 中文预训练 模型

为了处理中文，我们需要用transformers下载一个中文预训练模型。

Google BERT团队放出的官方中文预训练模型是一个很好的开始。这个模型涵盖了简体和繁体中文，12个隐层，768维嵌入向量，12头，1.1亿参数，大小约400MB。从如下的地址可以下载到它： [chinese_L-12_H-768_A-12.zip](https://link.zhihu.com/?target=https%3A//storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)。

但是我们现在并不需要手动去下载它，因为transformers已经帮我们打包好了这些共享模型，并且可以通过一致的from_pretrained接口下载模型到本地cache。

```python
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('bert-base-chinese')
```
第一次运行的时候，transformers会从s3.amazonaws.com 下载模型到本地一个cache目录，需要等待一段时间，后续运行就快了。

transformers模型管理的方式是为每一个模型起一个唯一的短名，如果一个模型同时有一个配套的tokenizer模型的话，它们会共用一个短名。上面提到的官方中文模型的短名就叫做“bert-base-chinese”。除了bert-base-chinese外，我们还可以找到clue/albert_chinese_small，voidful/albert_chinese_base等等几十个比较热门的社区贡献的中文预训练模型。下面这个网址可以找到所有社区共享由huggingface维护的模型列表。

[huggingface](https://huggingface.co/models)

同时，transformers也可以非常方便地直接使用其它途径下载到本地的各种预训练和微调模型。模型下载解压后，把上述短名参数换成本地模型路径名即可。

### 4. 向量编码和向量相似性展示

自然语言处理当中将一个词或者一句话编码成一个嵌入向量是最基础的操作，用transformers来编码非常简单。

```python
    # tensor([[ 101, 3217, 4697,  679, 6230, 3236,  102]])
    input_ids = tokenizer.encode('春眠不觉晓', return_tensors='pt')
    last_hidden_state, _ = model(input_ids) # shape (1, 7, 768) 
    v = torch.mean(last_hidden_stat, dim=1) # shape (1, 768)
```

注意到：

- model()实际上调用的是model.forward()函数
- model(input_ids)返回两个值last_hidden_state以及pooler_output。
  - last_hidden_state：shape=(1, 7, 768)即对输入的sequence中的每个token/字都返回了一个768维的向量。【因为 句子前面会加 [CLS] 后面 会加 [SEP]，所以长度变为 7】
  - pooler_output：序列中第一个特殊字符 [CLS] 对应的单个768维的向量，BERT模型会为单句的输入前后加两个特殊字符 [CLS] 和 [SEP]。
- 根据文档的说法，pooler_output 向量一般不是很好的句子语义摘要，因此这里采用了torch.mean对last_hidden_state 进行了求平均操作


得到了编码向量之后，就可以愉快地进行后续一系列的骚操作了。举个例子，我们可以比较以下四句诗词的语义相似性——“春眠不觉晓”、“大梦谁先觉”、“沉睡不消残酒”、“东临碣石以观沧海”。

```python
    import numpy as np

    def sentence_embedding(sentence):
        input_ids = tokenizer.encode(sentence, return_tensors='pt')
        hidden_states, _ = model(input_ids)
        return torch.mean(hidden_states, dim=1)

    sentences = ['春眠不觉晓','大梦谁先觉','浓睡不消残酒','东临碣石以观沧海']

    with torch.no_grad():
        vs = [sentence_embedding(sentence).numpy() for sentence in sentences]
        nvs = [v / np.linalg.norm(v) for v in vs] # normalize each vector
        m = np.array(nvs).squeeze(1) # shape (4, 768)
        print(np.around(m @ m.T, decimals=2)) # pairwise cosine similarity

    # Output
    [[1.   0.75 0.83 0.57]
    [0.75 1.   0.72 0.51]
    [0.83 0.72 1.   0.58]
    [0.57 0.51 0.58 1.  ]]
```