from common import model_utils
import torch.nn.utils.prune as prune

# only works on transformer layers
def magnitude(model, threshold):
    layers = model_utils.get_transformer_layers(model)
    opt = lambda x: 0 if abs(x) < threshold else x
    for name, params in layers.items():
        model_utils.map_weights_inplace(params, opt)