import torch
class Config():
  def __init__(self):
    self.split_ratio = 0.9   #训练和验证集的比例
    #MAX_SEQ_LEN = 50
    self.batch_size = 128
    self.seed = 0
    self.epochs = 10
    self.num_labels = 2
    self.is_colab = False
    if self.is_colab:
      self.data_index = 0
      self.is_demo = False
      self.data_path = "data/"
      self.bert_model_path = ".././../../bert_series/bert/bert-chinese-wwm_torch/"
    else:
      self.data_index = 0
      self.is_demo = False
      self.data_path = "F:/document/datasets/nlpData/sim/data/"
      self.bert_model_path = "F:/document/datasets/nlpData/pretrain/bert/chinese_wwm_ext_pytorch/"

    self.data_file_list = [
        "train","dev","test"
    ]
    self.col_list = ['sentence1','sentence2', 'label', 'label_id']
    self.sep = "\t"

    self.patience = 20
    self.do_lower_case = False
    self.device = torch.device('cpu' if not torch.cuda.is_available() else "cuda")
    
    # 模型参数
    self.no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    self.lr = 2e-05
    self.warmup = 0.1
    self.smoothing = 0.1

    # 模型地址
    self.f1_moadel_path = "checkpoint/checkpoint_f1.bin"
    self.loss_moadel_path = "checkpoint/checkpoint_loss.bin"
