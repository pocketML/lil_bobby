import torch

MODEL_INFO = {
    "base": {
        "path": "models/base",
        "download_url": "https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz"
    },
    "large": {
        "path": "models/large",
        "download_url": "https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz"
    }
}

# returns a dictionary with entries for all layers
# key is layer name, either encoding_layer, layer_i, or classification head name
# value is list of subcomponents (each a tuple of name and parameters)
def group_params_by_layer(model):
    layers = {}
    for name, param in model.named_parameters():
        if 'classification_heads' in name:
            key = name.split('classification_heads.')[1].split('.')[0]
        elif 'layers.' in name:
            key = 'layer_' + (name.split('layers.')[1].split('.')[0])
        else:
            key = 'encoding_layer'
        values = layers.get(key, [])
        values.append((name,param))
        layers[key] = values
    return layers

def get_transformer_layers(model):
    layers = group_params_by_layer(model)
    transformers = {k:v for k,v in layers.items() if 'layer_' in k}
    return transformers

# assuming weights are stored in 2-dimensional tensor
# opt is lambda that takes the old weight value and sets a new
def map_weights_inplace(layer, opt):
    for name, param in layer:
        if 'weight' in name:
            print(name, param.size())
            with torch.no_grad():
                size = list(param.size())
                for i in range(size[0]):
                    for j in range(size[1]):
                        value = param[i,j]
                        param[i][j] = opt(value)

def get_roberta_base(path):
    pass