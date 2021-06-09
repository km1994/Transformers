import os
import sys
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from tqdm import tqdm
from tqdm import tqdm_notebook, trange
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from torch.optim import optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
from tools import *
from Config import Config
from EarlyStopping import EarlyStopping
from DataPrecessForSingleSentence import DataPrecessForSingleSentence
from model import Bert_Classifier

if __name__=="__main__":
    config = Config()
    # 训练集加载
    train_data = load_data(config.data_path,config.data_file_list[config.data_index])
    # 标签映射表构建
    label2id_dic,id2label_dic = build_label2id(train_data,config.data_path)
    train_data = train_data[['text','label_id']]
    # 训练数据集拆分为训练集和验证集
    train, valid = train_test_split(train_data, train_size=config.split_ratio, random_state=config.seed)
    if config.is_demo:
        train = train[0:int(0.01*len(train))]
        valid = train[0:int(0.01*len(valid))]
    train_labels = train.groupby(['label_id']).count().reset_index()
    valid_labels = valid.groupby(['label_id']).count().reset_index()

    # 分词工具
    bert_tokenizer = BertTokenizer.from_pretrained(config.bert_model_path, do_lower_case=config.do_lower_case)
    # 类初始化
    processor = DataPrecessForSingleSentence(bert_tokenizer= bert_tokenizer)
    # 加载预训练的bert模型
    model = Bert_Classifier()
    # 数据加载器构建
    train_dataloder = build_data(processor,train,config.batch_size)
    valid_dataloder = build_data(processor,valid,config.batch_size)

    # 待优化的参数
    param_optimizer = list(model.named_parameters())
    
    optimizer_grouped_parameters = [
        {'params':[p for n, p in param_optimizer if not any(nd in n for nd in config.no_decay)], 'weight_decay':0.01},
        {'params':[p for n, p in param_optimizer if any(nd in n for nd in config.no_decay)], 'weight_decay':0.0}
    ]

    steps = len(train_dataloder) * config.epochs
    optimizer = BertAdam(
        optimizer_grouped_parameters, 
        lr=config.lr, 
        warmup= config.warmup , 
        t_total= steps
    )
    # 模型训练
    model = model.to(config.device)
    #存储loss
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)
    best_f1 = 0
    for i in trange(config.epochs, desc='Epoch'): 
        model.train() #训练
        for step, batch_data in enumerate(tqdm(train_dataloder, desc='Train Iteration')):
            batch_data = tuple(t.to(config.device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data
            # 对标签进行onehot编码
            if torch.cuda.is_available():
                one_hot = torch.zeros(batch_labels.size(0), config.num_labels).long().cuda()  #gpu版本
                one_hot_batch_labels = one_hot.scatter_(
                    dim=1,
                    index=torch.unsqueeze(batch_labels, dim=1),
                    src=torch.ones(batch_labels.size(0), config.num_labels
                ).long().cuda())
            else:
                # 以下注释为cpu版本
                one_hot = torch.zeros(batch_labels.size(0), config.num_labels).long()   #cpu版本
                one_hot_batch_labels = one_hot.scatter_(
                dim=1,
                index=torch.unsqueeze(batch_labels, dim=1),
                src=torch.ones(batch_labels.size(0), config.num_labels
                ).long())

            logits = model(
                batch_seqs, batch_seq_masks, batch_seq_segments
            )
            logits = torch.nn.functional.log_softmax(logits, dim=1)
            #loss_function = CrossEntropyLoss()
            loss = model.loss_fn(logits, batch_labels)
            loss.backward()
            train_losses.append(loss.item())
            print("\rloss : %f" % loss, end='')
            optimizer.step()
            optimizer.zero_grad()
            
        model.eval() #验证
        true_labels = []
        pred_labels = []
        for step, batch_data in enumerate(
                tqdm(valid_dataloder, desc='Dev Iteration')):
            with torch.no_grad():
                batch_data = tuple(t.to(config.device) for t in batch_data)
                batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data
                if torch.cuda.is_available():
                    # 对标签进行onehot编码，以下注释为gpu版本
                    one_hot = torch.zeros(batch_labels.size(0), config.num_labels).long().cuda()
                    one_hot_batch_labels = one_hot.scatter_(
                        dim=1,
                        index=torch.unsqueeze(batch_labels, dim=1),
                        src=torch.ones(batch_labels.size(0), config.num_labels
                    ).long().cuda())
                else:
                    # cpu
                    one_hot = torch.zeros(batch_labels.size(0), config.num_labels).long()
                    one_hot_batch_labels = one_hot.scatter_(
                    dim=1,
                    index=torch.unsqueeze(batch_labels, dim=1),
                    src=torch.ones(batch_labels.size(0), config.num_labels
                    ).long())

                logits = model(
                    batch_seqs, batch_seq_masks, batch_seq_segments
                )
                loss = model.loss_fn(logits, batch_labels)
                valid_losses.append(loss.item())
                
                predicts = model.predict(logits)
                pred_labels.append(predicts)
                true_labels.append(batch_labels.detach().cpu().numpy())
        
        true_labels = np.concatenate(true_labels)
        pred_labels = np.concatenate(pred_labels)
        precision = precision_score(true_labels, pred_labels, average='micro')
        recall = recall_score(true_labels, pred_labels, average='micro')
        f1 = f1_score(true_labels, pred_labels, average='micro')
        
        if best_f1<f1:
            torch.save(model, open(config.f1_moadel_path, "wb"))

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        print("train_loss:%f, valid_loss:%f, precision:%f, recall:%f, f1:%f " %(train_loss, valid_loss,precision,recall,f1))
        
        #重置训练损失和验证损失
        train_losses = []
        valid_losses = []
        
        early_stopping(valid_loss, model, config.loss_moadel_path)
        if early_stopping.early_stop:
            print("Early Stopping")
            break
    # 绘制 损失函数 loss 曲线
    draw_loss_pic(avg_train_losses,avg_valid_losses)