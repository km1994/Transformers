import os
import sys
import numpy as np
import random
from sklearn.metrics import classification_report
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from tqdm import tqdm_notebook, trange
import torch
from tools import *
from Config import Config
from DataPrecessForSingleSentence import DataPrecessForSingleSentence

if __name__=="__main__":
    config = Config()
    test_data = load_data(config.data_path,config.data_file_list[2])
    if config.is_demo:
        test_data = test_data.sample(int(0.01*len(test_data)))
    test_data = test_data[['text','label_id']]
    test_labels = test_data.groupby(['label_id']).count().reset_index()

    # 分词工具
    bert_tokenizer = BertTokenizer.from_pretrained(config.bert_model_path, do_lower_case=config.do_lower_case)
    # 类初始化
    processor = DataPrecessForSingleSentence(bert_tokenizer= bert_tokenizer)

    # 数据加载器构建
    test_dataloder = build_data(processor,test_data,config.batch_size)

    # 模型加载
    model = torch.load(config.f1_moadel_path)

    # 用于存储预测标签与真实标签
    true_labels = []
    pred_labels = []
    model.eval()
    # 预测
    with torch.no_grad():
        for batch_data in tqdm(test_dataloder, desc = 'TEST'):
            batch_data = tuple(t.to(config.device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data        
            logits = model(
                batch_seqs, batch_seq_masks, batch_seq_segments
            )
            predicts = model.predict(logits)
            pred_labels.append(predicts)
            true_labels.append(batch_labels.detach().cpu().numpy())

    # 查看各个类别的准确率和召回率
    result = classification_report(np.concatenate(true_labels), np.concatenate(pred_labels))
    print(result)
   
