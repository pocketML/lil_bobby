import copy

import torch
import torch.nn.utils.prune as prune

from common import model_utils
from embedding.base import Embedding

class MagnitudePruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold, module_to_prune=None):
        super().__init__()
        self.threshold = threshold
        self.module_to_prune = module_to_prune

    def compute_mask(self, inputs, default_mask):
        mask = default_mask.clone()
        mask[abs(inputs) < self.threshold] = 0
        return mask

class TopKPruning(prune.BasePruningMethod):
    """
    Prune the lowest ratio of parameters according to threshold.
    Threshold=0.3 would prune the weights with lowest value,
    pruning 30% of all weights in the model.
    """
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold, module_to_prune=None):
        super().__init__()
        self.threshold = threshold
        self.module_to_prune = module_to_prune

        if self.module_to_prune is not None:
            if isinstance(self.module_to_prune, torch.nn.LSTM):
                self.threshold *= 0.25
            elif isinstance(self.module_to_prune, torch.nn.Embedding):
                self.threshold *= 0.5

    def compute_mask(self, inputs, default_mask):
        mask = default_mask.clone()
        _, idx = torch.abs(inputs.flatten()).sort(descending=False)
        j = int(self.threshold * inputs.numel())

        # flat_out and mask access the same memory.
        flattened = mask.flatten()
        flattened[idx[:j]] = 0
        return mask

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
    grouped_params = model_utils.group_params_by_layer(model, "tang")

    parameters_to_prune = []

    for name in grouped_params:
        if True:#"embedding" not in name and "bilstm" not in name:
            module = getattr(model, name)

            is_sequential = isinstance(module, torch.nn.Sequential)
            is_embedding = isinstance(module, Embedding)

            if is_sequential:
                containers = module
            elif is_embedding:
                containers = [module.embedding]
            else:
                containers = [module]

            for container in containers:
                params = grouped_params[name] if not is_sequential else container.named_parameters()
                for param_name, param_values in params:
                    if "weight" in param_name:
                        name_fmt = param_name
                        if is_embedding:
                            name_fmt = ".".join(param_name.split(".")[2:])
                        elif not is_sequential:
                            name_fmt = ".".join(param_name.split(".")[1:])
                        parameters_to_prune.append((container, name_fmt, param_values))

    return parameters_to_prune

def prune_locally(module, name, values, prune_cls, threshold):
    pruning_instance = prune_cls(threshold, module_to_prune=module)
    mask = pruning_instance.compute_mask(values, torch.ones_like(values))
    prune.custom_from_mask(module, name, mask)

def prune_globally(params, prune_cls, threshold):
    prune.global_unstructured(
        [(x[0], x[1]) for x in params],
        pruning_method=prune_cls,
        threshold=threshold
    )

def prune_model(model, prune_cls, threshold, prune_local=False, sparsify=False):
    #model_copy = copy.deepcopy(model)

    params_to_prune = get_prunable_params(model)

    if prune_local:
        for module, name, values in params_to_prune:
            prune_locally(module, name, values, prune_cls, threshold)
    else:
        prune_globally(params_to_prune, prune_cls, threshold)

    for module, param_name, _ in params_to_prune:
        prune.remove(module, param_name)
        if sparsify:
            dense_tensor = getattr(module, param_name)
            sparse_tensor = torch.nn.Parameter(to_sparse(dense_tensor))
            setattr(module, param_name, sparse_tensor)

    return model

def params_zero(model):
    zero = 0
    total_params = 0
    for _, param in model.named_parameters():
        non_zero = torch.count_nonzero(param).item()
        num_params = param.size().numel()
        zero += (num_params - non_zero)
        total_params += num_params
    return total_params, zero
