import torch
import torch.nn as nn
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from LabelSmoothing import LabelSmoothing
from Config import Config
args = Config()

class Bert_Classifier(nn.Module):
    def __init__(self):
        super(Bert_Classifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            args.bert_model_path, 
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
            num_labels=args.num_labels
        )
        self.loss_function = LabelSmoothing(args.num_labels, args.smoothing)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        label_id=None,
        output_all_encoded_layers=False
    ):
        logits = self.bert(input_ids, token_type_ids)
        logits = torch.nn.functional.log_softmax(logits, dim=1)
        return logits
    
    # 功能：计算 loss 
    def loss_fn(self, bert_encode, labels):
        loss = self.loss_function(bert_encode, labels)
        return loss

    # 功能：预测
    def predict(self, bert_encode):
        bert_encode = bert_encode.softmax(dim=1).argmax(dim = 1)
        predicts = bert_encode.detach().cpu().numpy()
        return predicts

    # 功能：计算 acc 和 f1
    def acc_f1(self, y_pred, y_true):
        pass