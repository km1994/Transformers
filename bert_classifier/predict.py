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
    query_list = [
        "25日股票基金全线受挫 九成半基金跌逾1%全景网3月26日讯 周三开放式基金净值普降，股票型基金全线受挫，九成半基金跌幅超过1%。上证综指昨日收市跌2.00%。据全景网基金统计数据，3月25日统计的229只股票型基金全线下挫，其中219只跌幅在1%以上，占比95.63%。跌幅排名前五位的基金是友邦红利ETF、富国天鼎、宝盈资源优选、德盛红利、易方达深100ETF，增长率分别为-3.03%、-2.70%、-2.51%、-2.44%、-2.42%。华富策略精选、景顺公司治理、长城双动力、荷银合丰稳定、汇丰晋信2026等跌幅较小，均在1%以内。积极配置型基金亦全盘尽墨，中欧新蓝筹、金鹰优选两只基金跌幅超过2%，天治财富增长、华夏回报跌幅在0.5%以内。保守配置型基金中，申万盛利配置最为抗跌，下挫0.20%，国投瑞银融华垫底，跌0.99%。债市方面，上证国债指数昨日涨0.03%，上证企债指数跌0.10%。普通债券型基金仅4只飘红，国投瑞银债券领跑，涨0.03%，中信稳定双利、易方达稳健收益B、易方达稳健收益A也小幅上扬。嘉实多元收益A、嘉实多元收益B、华富收益增强B、华富收益增强A跌幅都超过0.4%。（全景网/陈丹蓉）",
        "古天乐投诉大S不给机会 希望下次再合作(组图)新浪娱乐讯 8月26日，电影《保持通话》举行宣传活动，众演员包括古天乐( 听歌 blog)、徐熙媛(大S)、张嘉辉及导演陈木胜等等均有出席。古天乐笑言这次拍摄多是单独演出，并要一直拿着手机演戏，感觉新鲜又甚具压力，他更表示与大S只有一场对手戏，希望下次能够再有机会合作。TUNGSTAR/文并图 "
    ]
    # predict_data = pd.DataFrame([{'text':query, 'label_id':0} for query in query_list])
    id2label_dic = load_plk_dict(config.data_path,"label2id")['id2label']

    # 分词工具
    bert_tokenizer = BertTokenizer.from_pretrained(config.bert_model_path, do_lower_case=config.do_lower_case)
    # 类初始化
    processor = DataPrecessForSingleSentence(bert_tokenizer= bert_tokenizer)

    # 模型加载
    model = torch.load(config.f1_moadel_path)

    model.eval()
    pred_labels = []

    # 预测
    while True:
        # 输入句子
        query = input("输入句子：")
        if len(query)>0:
            predict_data = pd.DataFrame([{'text':query, 'label_id':0}])
            # 数据加载器构建
            predict_dataloder = build_data(processor,predict_data,config.batch_size)
            # 预测
            with torch.no_grad():
                pred_labels = []
                for batch_data in tqdm(predict_dataloder, desc = 'Predict'):
                    batch_data = tuple(t.to(config.device) for t in batch_data)
                    batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data        
                    logits = model(
                        batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
                    logits = logits.softmax(dim=1).argmax(dim = 1)
                    pred_labels = logits.detach().cpu().numpy()

                print(f"pred_labels:{pred_labels}")
                print(f"pred_labels:{id2label_dic[pred_labels[0]]}")
    
