from common import model_utils
import torch.nn.utils.prune as prune

# only works on transformer layers
def magnitude(model, threshold):
    layers = model_utils.get_transformer_layers(model)
    pass
    # do some pytorch builtin pruning?