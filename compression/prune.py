from abc import abstractmethod
import torch
from common import model_utils
import torch.nn.utils.prune as prune
import copy

class MagnitudePruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def compute_mask(self, t, default_mask):
        print(self._tensor_name)
        mask = default_mask.clone()
        mask[abs(t) < self.threshold] = 0
        return mask

class MovementAutogradWrappper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, threshold):
        mask = inputs.clone()
        _, idx = inputs.flatten().sort(descending=True)
        j = int(threshold * inputs.numel())

        # flat_out and mask access the same memory.
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0
        flat_out[idx[:j]] = 1
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class MovementPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def compute_mask(self, inputs, default_mask):
        mask = inputs.clone()
        _, idx = inputs.flatten().sort(descending=True)
        j = int(self.threshold * inputs.numel())

        # flat_out and mask access the same memory.
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0
        flat_out[idx[:j]] = 1
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
        if "embedding" not in name and "mask" not in name:
            module = getattr(model, name)

            is_sequential = isinstance(module, torch.nn.Sequential)

            if is_sequential:
                containers = module
            else:
                containers = [module]

            for container in containers:
                params = container.named_parameters() if is_sequential else grouped_params[name]
                for param_name, param_values in params:
                    if "weight" in param_name:
                        name_fmt = param_name
                        if not is_sequential:
                            name_fmt = ".".join(param_name.split(".")[1:])
                        parameters_to_prune.append((container, name_fmt, param_values))

    return parameters_to_prune

def do_pruning(params, prune_cls, importance_scores=None, sparsify=False, **kwargs):
    prune.global_unstructured(
        [(x[0], x[1]) for x in params],
        pruning_method=prune_cls,
        importance_scores=importance_scores,
        **kwargs
    )

    for module, param_name, _ in params:
        if sparsify:
            dense_tensor = getattr(module, param_name)
            sparse_tensor = torch.nn.Parameter(to_sparse(dense_tensor))
            setattr(module, param_name, sparse_tensor)

def ratio_zero(model):
    zero = 0
    total_params = 0
    for name, param in model.named_parameters():
        if "mask" not in name:
            non_zero = torch.count_nonzero(param).item()
            num_params = param.size().numel()
            zero += (num_params - non_zero)
            total_params += num_params
    return zero / total_params

def magnitude_pruning(model, threshold):
    model = copy.deepcopy(model)

    params_to_prune = get_prunable_params(model)

    do_pruning(params_to_prune, MagnitudePruning, threshold=threshold)

    for module, param_name in params_to_prune:
        prune.remove(module, param_name)

    return model

def get_mask_scores(model):
    grouped_params = model_utils.group_params_by_layer(model, "tang")

    masks_dict = {}

    for name in grouped_params:
        if "mask" in name:
            module = getattr(model, name)

            is_sequential = isinstance(module, torch.nn.Sequential)

            if is_sequential:
                containers = module
            else:
                containers = [module]

            for container in containers:
                params = container.named_parameters() if is_sequential else grouped_params[name]
                for param_name, param_mask in params:
                    if "weight" in param_name:
                        masks_dict[(container, param_name)] = param_mask

    return masks_dict

def movement_pruning(model, threshold):
    model = copy.deepcopy(model)

    params_to_prune = get_prunable_params(model)
    mask_scores = get_mask_scores(model)

    do_pruning(params_to_prune, MovementPruning, importance_scores=mask_scores, threshold=threshold)
    for module, name in mask_scores:
        setattr(model, name, mask_scores[(module, name)])

    return model

def initalize_mask_scores(model):
    for m, n, p in get_prunable_params(model):
        print(n)
        mask_scores = torch.nn.Parameter(torch.Tensor(p.size()))
        torch.nn.init.constant(mask_scores, val=0.0)
        model.register_parameter(f"{n}_mask", mask_scores)
