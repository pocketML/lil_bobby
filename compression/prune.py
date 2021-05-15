import copy

import torch
import torch.nn.utils.prune as prune

from common import model_utils
from embedding.base import Embedding

class MagnitudePruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def compute_mask(self, inputs, default_mask):
        mask = default_mask.clone()
        mask[abs(inputs) < self.threshold] = 0
        return mask

class TopKPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def compute_mask(self, inputs, default_mask):
        mask = default_mask.clone()
        _, idx = torch.abs(inputs.flatten()).sort(descending=False)
        j = int(self.threshold * inputs.numel())

        # flat_out and mask access the same memory.
        flattened = mask.flatten()
        flattened[idx[:j]] = 0
        return flattened

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def to_dense(x):
    return x.to_dense()

def get_prunable_params(model):
    grouped_params = model_utils.group_params_by_layer(model, None)
    parameters_to_prune = []

    for name in grouped_params:
        module = getattr(model, name)

        is_sequential = isinstance(module, torch.nn.Sequential)
        is_embedding = isinstance(module, Embedding)

        if is_sequential:
            containers = module
        elif is_embedding:
            if module.embedding is not None:
                containers = [module.embedding]
            else:
                containers = [module.vectors, module.scalars]
        else:
            containers = [module]

        for container in containers:
            params = container.named_parameters() if is_sequential or is_embedding else grouped_params[name]
            for param_name, _ in params:
                if "weight" in param_name:
                    name_fmt = param_name
                    if not is_sequential and not is_embedding:
                        name_fmt = ".".join(param_name.split(".")[1:])
                    parameters_to_prune.append((container, name_fmt))

    return parameters_to_prune

def do_pruning(params, prune_cls, sparsify=False, **kwargs):
    prune.global_unstructured(
        params,
        pruning_method=prune_cls,
        **kwargs
    )

    for module, param_name in params:
        prune.remove(module, param_name)
        if sparsify:
            dense_tensor = getattr(module, param_name)
            sparse_tensor = torch.nn.Parameter(to_sparse(dense_tensor))
            setattr(module, param_name, sparse_tensor)

def params_zero(model):
    zero = 0
    total_params = 0
    for _, param in model.named_parameters():
        non_zero = torch.count_nonzero(param).item()
        num_params = param.size().numel()
        zero += (num_params - non_zero)
        total_params += num_params
    return total_params, zero

def magnitude_pruning(model, threshold):
    model_copy = copy.deepcopy(model)

    params_to_prune = get_prunable_params(model_copy)

    do_pruning(params_to_prune, MagnitudePruning, threshold=threshold)

    return model_copy

def movement_pruning(model, threshold):
    """
    Not implemented. Simply return model unchanged.
    """
    return model

def topk_pruning(model, threshold):
    """
    Prune the lowest ratio of parameters according to threshold.
    Threshold=0.3 would prune the weights with lowest value,
    pruning 30% of all weights in the model.
    """
    model_copy = copy.deepcopy(model)

    params_to_prune = get_prunable_params(model_copy)

    do_pruning(params_to_prune, TopKPruning, threshold=threshold)

    return model_copy
