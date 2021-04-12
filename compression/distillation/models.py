
from compression.distillation.student_models.tang_bilstm import TangBILSTM
from compression.distillation.student_models.glue_bilstm import GlueBILSTM
from compression.distillation.student_models.wasserblat_ffn import WASSERBLAT_FFN
from compression.distillation.student_models.base import get_dist_loss_function as DistLossFunction
from compression.distillation.student_models import base

STUDENT_MODELS = {
    'glue' : GlueBILSTM,
    'wasserblat-ffn': WASSERBLAT_FFN,
    'tang': TangBILSTM
}

def load_student(task, student_type, use_gpu, model_name=None):
    cfg = base.get_default_student_config(task, student_type, use_gpu=use_gpu, model_name=model_name)
    try:
        model = STUDENT_MODELS[student_type](cfg)
    except KeyError:
        raise Exception(f'Student type "{student_type}" not recognized')
    # load state dict
    if model_name is not None:
        model.load(model_name)
    return model
