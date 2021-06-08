import pickle
# 功能：字典数据 存储 为 plk
def save_plk_dict(dic,save_path,fila_name):
    '''
        功能：字典数据 存储 为 plk
        input:
        dic          Dict     存储字典    
        save_path     String    存储目录 
        fila_name     String    存储文件 
    return:
        
    '''
    with open(save_path+ fila_name + '.pkl', 'wb') as f:
        pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL) 

def load_plk_dict(save_path,fila_name):
    '''
        功能：加载 plk 中 字典数据
        input: 
        save_path     String    存储目录 
        fila_name     String    存储文件 
    return:
        dic        Dict     字典数据 
        
    '''
    with open(save_path+ fila_name + '.pkl', 'rb') as f:
        return pickle.load(f)     

import pandas as pd
from sklearn.preprocessing import LabelEncoder
# 功能：数据加载函数
def load_data(data_path,data_file,col_list=['sentence1','sentence2', 'label', 'label_id'], sep=","):
    '''
        功能：数据加载函数
        input:
            data_path     String      数据目录   
            data_file     String      数据文件名称
        return:
            data        List       数据
            num_labels     int       标签类别数量
    '''
    # 数据加载
    data = pd.read_table(f'{data_path}{data_file}.txt', encoding='utf-8', names=col_list[0:-1],sep=sep)
    # 标签编码
    le = LabelEncoder()
    le.fit(data.label.tolist())
    data[col_list[-1]] = le.transform(data.label.tolist())
    return data[col_list] 

# 功能：标签映射表构建
def build_label2id(data,data_path):
    '''
        功能：标签映射表构建
        input:
            data     DataFrame     训练数据
        return:
            label2id_dic Dict        label 到 id 的映射
            id2label_dic Dict        id 到 label 的映射
    '''
    labeldata = data.groupby(['label', 'label_id']).count().reset_index()
    label2id_dic = {}
    id2label_dic = {}
    for index,row in labeldata.T.iteritems():
        label2id_dic[row['label']] = row['label_id']
        id2label_dic[row['label_id']] = row['label']

    label2id = {
        "label2id":label2id_dic,
        "id2label":id2label_dic
    }

    save_plk_dict(label2id,data_path,"label2id")
    return label2id_dic,id2label_dic

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
# 功能：数据加载器构建
def build_data(processor,data,batch_size):
    '''
        功能：数据加载器构建
    '''
    # 产生训练集输入数据
    seqs, seq_masks, seq_segments, labels = processor.get_input(dataset=data)
    # 转换为torch tensor
    t_seqs = torch.tensor(seqs, dtype=torch.long)
    t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
    t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
    t_labels = torch.tensor(labels, dtype = torch.long)

    data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)
    sampler = RandomSampler(data)
    dataloder = DataLoader(dataset= data, sampler=sampler, batch_size = batch_size)
    return dataloder

import matplotlib.pyplot as plt
# 功能：绘制 损失函数 loss 曲线
def draw_loss_pic(avg_train_losses,avg_valid_losses):
    '''
        功能：绘制 损失函数 loss 曲线
        input:
            avg_train_losses
            avg_valid_losses
    '''
    fig = plt.figure(figsize=(8,6))
    plt.plot(range(1, len(avg_train_losses)+1), avg_train_losses, label='Training Loss')
    plt.plot(range(1, len(avg_valid_losses)+1), avg_valid_losses, label='Validation Loss')

    #find the position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses))+1
    plt.axvline(minposs, linestyle='--', color = 'r', label='Early Stopping Checkpoint')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')