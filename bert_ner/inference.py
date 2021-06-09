import os
from net.bert_ner import Bert_CRF
from data_processor import convert_query_to_features
import torch
from Config import Config
args = Config()
from util.model_util import load_model
from util.score import get_tags
from util.metrics import get_chunk
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
tokenizer = BertTokenizer.from_pretrained(
    args.bert_model, 
    cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
    do_lower_case=args.do_lower_case
)
def start():
    # produce_data()
    model = Bert_CRF()
    model.load_state_dict(load_model(args.output_dir))
    model.to(args.device)

    # -----------------------验证----------------------------
    model.eval()

    with torch.no_grad():
        while True:
            query = input("输入query：")
            print(f"\nquery:{query}")
            input_ids,input_mask,segment_ids,output_mask = convert_query_to_features(query, args.max_seq_length, tokenizer)
            all_input_ids = torch.tensor([input_ids], dtype=torch.long)
            all_input_mask = torch.tensor([input_mask], dtype=torch.long)
            all_segment_ids = torch.tensor([segment_ids], dtype=torch.long)
            all_output_mask = torch.tensor([output_mask], dtype=torch.long)
            if torch.cuda.is_available():
                bert_encode = model(all_input_ids, all_input_mask, all_segment_ids)
                predicts = model.predict(bert_encode, all_output_mask)
            else:
                bert_encode = model(all_input_ids, all_input_mask, all_segment_ids).cpu()
                predicts = model.predict(bert_encode, all_output_mask).cpu()

            predicts = predicts.view(1, -1).squeeze()
            predicts = predicts[predicts != -1]
            pre_tags = get_tags(predicts.cpu().numpy().tolist(),args.labels)
            print(f'pre_tags:{pre_tags}')

            entity = get_chunk(predicts.cpu().numpy().tolist(),args.tag_map,query)
            print(f'entity:{entity}')

if __name__ == '__main__':
    start()
