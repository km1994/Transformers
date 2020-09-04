# 【HugBert03】最后一英里：下游NLP任务的微调

> 作者：杨夕
> 
> github:https://github.com/km1994/
> 
> 参考：https://zhuanlan.zhihu.com/p/149779660

## 前言

终于挤出点时间，接着写HugBert系列第三篇，介绍如何用transformers来对下游NLP任务进行微调（fine tune）。

预训练语言模型的范式就是在大数据集上让模型习得通用的语言知识，并在此基础上，针对目标任务进行小规模的微调，达到知识迁移、算力共享和专项高精等多赢的目标。Huggingface Transformers整合了类BERT模型的接口，提供了社区分享的预训练模型库，铺平了Pytorch和Tensorflow之间互操作的鸿沟，但我们还需要完成最后一英里任务：微调。

## 1 微博评论情感分析任务



