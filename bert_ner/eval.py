import os
from net.bert_ner import Bert_CRF
from data_loader import create_batch_iter
import torch
from util.model_util import load_model

from util.score import get_tags
from Config import Config
args = Config()

def start():
    model = Bert_CRF()
    model.load_state_dict(load_model(args.output_dir))
    model.to(args.device)
    print('create_iter')
    eval_iter = create_batch_iter("valid")
    print('create_iter finished')

    # -----------------------验证----------------------------
    model.eval()

    y_predicts, y_labels = [], []
    with torch.no_grad():
        for step, batch in enumerate(eval_iter):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, output_mask = batch
            bert_encode = model(input_ids, segment_ids, input_mask).cpu()
            predicts = model.predict(bert_encode, output_mask).cpu()

            label_ids = label_ids.view(1, -1).squeeze()
            predicts = predicts.view(1, -1).squeeze()
            label_ids = label_ids[label_ids != -1]
            predicts = predicts[predicts != -1]

            y_predicts.append(predicts)
            y_labels.append(label_ids)
        
        eval_predicted = torch.cat(y_predicts, dim=0).cpu()
        eval_labeled = torch.cat(y_labels, dim=0).cpu()
        eval_acc = cul_acc(eval_predicted,eval_labeled)

        eval_recalls = []
        eval_precisions = []
        eval_f1s = []
        for predict,label_id in zip(y_predicts,y_labels):
            f1, precision, recall  = gen_metrics(
                label_id, predict, args.tag_map
            )
            eval_recalls.append(recall)
            eval_precisions.append(precision)
            eval_f1s.append(f1)

        eval_precision = mean(eval_precisions)
        eval_recall = mean(eval_recalls)
        eval_f1 = mean(eval_f1s)

        info = 'eval_acc:%4f - p:%4f - r:%4f - f1:%4f\n' % (
                   eval_acc,
                   eval_precision, 
                   eval_recall, 
                   eval_f1
                )
        print(f"info=>{info}")

# 功能：计算 acc
def cul_acc(y_true,y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    correct = np.sum((y_true == y_pred).astype(int))
    acc = correct / y_pred.shape[0]
    return acc      

if __name__ == '__main__':
    start()
