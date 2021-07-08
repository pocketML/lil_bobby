from copy import deepcopy
import torch
import torch.nn.utils.prune as prune

from common import model_utils, data_utils
from embedding.base import Embedding
import evaluate
from analysis.parameters import get_theoretical_size, get_model_sparsity
from analysis import pretty_print

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
                self.threshold *= 1.5
            elif isinstance(self.module_to_prune, torch.nn.Embedding):
                self.threshold *= 1.5
            elif isinstance(self.module_to_prune, torch.nn.Linear):
                self.threshold *= 0.5

    def compute_mask(self, inputs, default_mask):
        mask = default_mask.clone()
        _, idx = torch.abs(inputs.flatten()).sort(descending=False)
        j = int(self.threshold * inputs.numel())

        # flattened and mask access the same memory.
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
            for param_name, param_values in params:
                if "weight" in param_name or "bias" in param_name:
                    name_fmt = param_name
                    if not is_sequential and not is_embedding:
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

def actual_pruning(model, prune_cls, threshold, prune_local=False, sparsify=False):
    params_to_prune = get_prunable_params(model)

    if prune_local:
        for module, name, values in params_to_prune:
            prune_locally(module, name, values, prune_cls, threshold)
    else:
        prune_globally(params_to_prune, prune_cls, threshold)

    for module, param_name, _ in params_to_prune:
        prune.remove(module, param_name)
        if sparsify: # converts pruned tensors to sparse tensors
            dense_tensor = getattr(module, param_name)
            sparse_tensor = torch.nn.Parameter(to_sparse(dense_tensor))
            setattr(module, param_name, sparse_tensor)

    return model

def do_pruning(model, args, epoch=None):
    pruned_model = deepcopy(model)

    threshold = args.prune_threshold
    if epoch is not None and epoch < args.prune_warmup:
        threshold = threshold * (epoch / args.prune_warmup)

    prune_class = None
    if args.prune_magnitude:
        prune_class = MagnitudePruning
    elif args.prune_movement:
        pass
    elif args.prune_topk:
        prune_class = TopKPruning

    pruned_model = actual_pruning(pruned_model, prune_class, threshold, args.prune_local)

    params, zero = get_model_sparsity(pruned_model)
    sparsity = (zero / params) * 100
    print(f"Sparsity: {sparsity:.2f}%")

    return pruned_model

def prune_model(model, device, args, sacred_experiment=None):
    dl = data_utils.get_val_dataloader(model, data_utils.load_val_data(args.task))

    print("Starting point:")
    pretty_print.print_model_disk_size(model, sacred_experiment)
    params, zero = get_model_sparsity(model)
    sparsity = (zero / params) * 100
    print(f"Sparsity before: {sparsity:.2f}%")
    evaluate.evaluate_distilled_model(model, dl, device, args)
    print()

    print("** Pruning model... **")
    model = do_pruning(model, args)

    nonzero_params, nonzero_bits = get_theoretical_size(model, sacred_experiment)
    print(f"Non-zero params: {nonzero_params}")
    print(f"Theoretical size: {nonzero_bits:.3f} MB")
    pretty_print.print_model_disk_size(model, sacred_experiment)

    print("** Pruning completed **")
    evaluate.evaluate_distilled_model(model, dl, device, args)
    print()
    return model
