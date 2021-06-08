import os
import torch
from net.bert_ner import Bert_CRF
from Config import Config
args = Config()

def save_model(model, output_dir):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)


def load_model(output_dir):
    # Load a trained model that you have fine-tuned
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    if torch.cuda.is_available():
        model_state_dict = torch.load(output_model_file)
    else:
        model_state_dict = torch.load(output_model_file,map_location='cpu')
    return model_state_dict
