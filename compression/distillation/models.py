
from compression.distillation.student_models.tang_bilstm import TangBILSTM
from compression.distillation.student_models.glue_bilstm import GlueBILSTM
from compression.distillation.student_models.wasserblat_ffn import WASSERBLAT_FFN
from compression.distillation.student_models.char_rnn import CharRNN
from compression.distillation.student_models.transformer import Transformer
from compression.distillation.student_models.base import get_dist_loss_function as DistLossFunction
from compression.distillation.student_models import base

STUDENT_MODELS = {
    'glue' : GlueBILSTM,
    'wasserblat-ffn': WASSERBLAT_FFN,
    'tang': TangBILSTM,
    'char-rnn': CharRNN,
    'transformer': Transformer
}

def load_student(task, student_type, use_gpu, model_name=None):
    cfg = base.get_default_student_config(task, student_type, model_name=model_name)
    if student_type in STUDENT_MODELS:
        model = STUDENT_MODELS[student_type](cfg)
    else:
        raise Exception(f'Student type "{student_type}" not recognized')
    # load state dict
    if model_name is not None:
        model.load(model_name)
    model.cfg['use-gpu'] = use_gpu # very important part
    return model
