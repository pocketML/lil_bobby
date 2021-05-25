
from compression.distillation.student_models.bilstm import BiLSTM
from compression.distillation.student_models.rnn import RNN
from compression.distillation.student_models.transformer import Transformer
from compression.distillation.student_models.transformer2 import Transformer2
from compression.distillation.student_models.transformer3 import Transformer3
from compression.distillation.student_models.emb_ffn import EmbFFN
from compression.distillation.student_models import base

STUDENT_MODELS = {
    'glue': None,
    'bilstm': BiLSTM,
    'rnn': RNN,
    'transformer': Transformer,
    'trans2': Transformer2,
    'trans3': Transformer3,
    'emb-ffn': EmbFFN
}

def load_student(task, student_type, use_gpu, model_name=None, args=None):
    cfg = base.get_default_student_config(task, student_type, model_name=model_name)
    if args is not None:
        args_dict = args.__dict__
        for key in args_dict:
            if args_dict[key] is not None:
                cfg[key.replace("_", "-")] = args_dict[key]

    if student_type in STUDENT_MODELS:
        model = STUDENT_MODELS[student_type](cfg)
    else:
        raise Exception(f'Student type "{student_type}" not recognized')
    # load state dict
    if model_name is not None:
        model.load(model_name)
    model.cfg['use-gpu'] = use_gpu # very important part
    return model
