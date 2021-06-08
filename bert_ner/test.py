import os
from net.bert_ner import Bert_CRF
from data_loader import create_inference_iter
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
    eval_iter = create_inference_iter()
    print('create_iter finished')

    # -----------------------验证----------------------------
    model.eval()

    y_predicts = []
    with torch.no_grad():
        for step, batch_data in enumerate(eval_iter):
            text_list, batch = batch_data
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, output_mask = batch
            bert_encode = model(input_ids, segment_ids, input_mask).cpu()
            predicts = model.predict(bert_encode, output_mask).cpu()

            predicts = predicts.view(1, -1).squeeze()
            predicts = predicts[predicts != -1]
            pre_tags = get_tags(predicts.cpu().numpy().tolist(),args.labels)
            # print(f'pre_tags:{pre_tags}')
            # pre_entities = [text[tag[0]:tag[1] + 1] for tag in pre_tags]
            # print(f'pre:{pre_entities}')

            # y_predicts.append(predicts)
        
        # print(f"len(y_predicts):{len(y_predicts)}")
        # print(f"y_predicts:{y_predicts}")

if __name__ == '__main__':
    start()
