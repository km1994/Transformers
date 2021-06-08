import torch
class Config():
    def __init__(self):
        self.STOP_WORD_LIST = None
        self.CUSTOM_VOCAB_FILE = None
        self.VOCAB_FILE = "model/vocab.txt"
        self.log_path = "output/logs"
        self.plot_path = "output/images/loss_acc.png"
        self.cache_dir = "model/"
        self.output_dir = "output/checkpoint"  # checkpoint和预测输出文件夹
        self.platform = "colab"
        if self.platform == "colab":
            self.bert_model = ".././../../bert_series/bert/bert-chinese-wwm_torch/"  
        elif self.platform == "local":
            self.bert_model = "F:/document/datasets/nlpData/pretrain/bert/chinese_wwm_ext_pytorch/"  
        elif self.platform == "atlas":
            self.bert_model = "/home/yangkm/pretrain/bert/chinese_wwm_ext_pytorch/"  

        self.sep = " "
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

        self.task_name = "bert_ner"  # 训练任务名称
        self.bert_cache = '/cache/'
        self.flag_words = ["[PAD]", "[CLP]", "[SEP]", "[UNK]"]
        self.max_seq_length = 220
        self.do_lower_case = True
        self.train_batch_size = 16
        self.eval_batch_size = 16
        self.inference_batch_size = 16
        self.learning_rate = 2e-5
        self.num_train_epochs = 20
        self.warmup_proportion = 0.1
        self.no_cuda = False
        self.device = "cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu"
        self.seed = 233
        self.gradient_accumulation_steps = 1
        self.fp16 = False
        self.loss_scale = 0.
        self.train_size = 0.7
        self.no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
