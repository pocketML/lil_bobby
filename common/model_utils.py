import torch.nn as nn

import os
from fairseq.models.roberta import RobertaModel

from common import task_utils
from preprocessing import download

# roberta models
MODEL_INFO = {
    "base": {
        "path": "models/roberta.base",
        "download_url": ["https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz"]
    },
    "large": {
        "path": "models/roberta.large",
        "download_url": ["https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz"]
    }
}

# returns a dictionary with entries for all layers
# key is layer name, either encoding_layer, layer_i, or classification head name
# value is list of subcomponents (each a tuple of name and parameters)
def group_params_by_layer(model, arch):
    layers = {}
    for name, param in model.named_parameters():
        if arch in MODEL_INFO.keys():
            if 'classification_heads' in name:
                key = name.split('classification_heads.')[1].split('.')[0]
            elif 'layers.' in name:
                key = 'layer_' + (name.split('layers.')[1].split('.')[0])
            elif 'sentence_encoder' in name:
                key = 'sentence_encoder'
            else:
                key = 'lm_head'
        else: # we have a student model
            key = name.split('.')[0]
        values = layers.get(key, [])
        values.append((name,param))
        layers[key] = values
    return layers

def get_transformer_layers(model):
    layers = group_params_by_layer(model)
    transformers = {k:v for k,v in layers.items() if 'layer_' in k}
    return transformers

def get_model_path(task, model_type):
    """
    Returns a path to where a type of model is saved.
    F.x. models/finetuned/sst-2.
    """
    if model_type not in ("finetuned", "distilled", "embeddings"):
        raise ValueError("Invalid model type.")

    model_path = f"models/{model_type}/{task}"

    #hpc_shared_path = "/home/data_shares/lil_bobby"
    # Disabled for now cause of permission stuff
    # if os.path.exists(hpc_shared_path):
    #     model_path = f"{hpc_shared_path}/{model_path}"

    os.makedirs(model_path, exist_ok=True)
    return model_path

def load_teacher(task, checkpoint_path, use_cpu=False, model_name='checkpoint_best.pt'):
    bin_path = task_utils.get_processed_path(task)
    model = RobertaModel.from_pretrained(
        checkpoint_path,
        checkpoint_file=model_name,
        data_name_or_path=bin_path
    )
    model.eval()
    if not use_cpu:
        model.cuda()
    return model

def load_roberta_model(arch, use_cpu=False):
    model_info = MODEL_INFO['base'] if arch == 'roberta_base' else MODEL_INFO['large']
    model_dir = download.get_roberta_path(model_info)
    model = RobertaModel.from_pretrained(model_dir, checkpoint_file='model.pt')
    if not use_cpu:
        model.cuda()
    model.eval()
    return model

def is_finetuned_model(arch):
    return arch in MODEL_INFO.keys()

def is_quantized_model(model):
    for module in model._modules:
        if "quant" in str(model._modules[module]).lower():
            return True
    return False

class GlueBaseline(nn.Module):
  def __init__(self, task, vocab_size=2200000):
    super().__init__()

    self.emb = nn.Embedding(vocab_size, 300)
    self.bilstm = nn.LSTM(300, 1500, 2, bidirectional=True)
    cls_in = 1500 * 2
    if task in ("qqp", "mnli"):
        cls_in *= 4 # GLUE uses cat-cmp with sentence pair tasks.

    self.cls = nn.Sequential( # their fancy MLP version
      nn.Linear(1500 * 2 * 4, 512),
      nn.Linear(512, 512),
      nn.Linear(512, 2)
    )

  def forward(self, x):
    return self.bilstm(x)