# 【关于 transformers 管道(pipeline)】那些你不知道的事

> 作者：杨夕
> 
> github:https://github.com/km1994/
> 
> 翻译地址：https://huggingface.co/transformers/quicktour.html 
> 
> 参考：https://blog.csdn.net/CY19980216/article/details/114292515


## 一、开始使用 管道(pipeline) 执行任务

### 1.1 常见的 NLP 任务类型

- (1) 情感分析(Sentiment analysis): 判断文本是积极的或是消极的;
- (2) 文本生成(Text generation): 根据某种提示生成一段相关文本;
- (3) 命名实体识别(Name entity recognition): 判断语句中的某个分词属于何种类型;
- (4) 问答系统(Question answering): 根据上下文和问题生成答案;
- (5) 缺失文本填充(Filling masked text): 还原被挖去某些单词的语句;
- (6) 文本综述(Summarization): 根据长文本生成总结性的文字;
- (7) 机器翻译(Translation): 将某种语言的文本翻译成另一种语言;
- (8)特征挖掘(Feature extraction): 生成文本的张量表示;

## 二、使用 pipeline 处理 NLP 任务

### 2.1 情感分析任务





