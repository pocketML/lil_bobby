from fairseq.models.roberta import RobertaModel
from common.task_utils import TASK_INFO
import download

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

def load_teacher(task, use_cpu=False):
    bin_path = f'{TASK_INFO[task]["path"]}/processed/{task}-bin/'
    model = RobertaModel.from_pretrained(
        "checkpoints", #f'models/experiments/finetune_{task}',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=bin_path
    )
    model.eval()
    if not use_cpu:
        model.cuda()
    return model

def load_roberta_model(arch, use_cpu=False):
    model_dir = download.get_roberta_path('base' if arch == 'roberta_base' else 'large')
    model = RobertaModel.from_pretrained(model_dir, checkpoint_file='model.pt')
    if not use_cpu:
        model.cuda()
    model.eval()
    return model

def load_student(task, student_type, use_gpu, load_saved_model=None):
    cfg = base.get_default_config(task, student_type, use_gpu=use_gpu, model_name=load_saved_model)
    try:
        model = STUDENT_MODELS[student_type](cfg)
    except:
        raise Exception(f'Student type "{student_type}" not recognized')
    # load state dict
    if load_saved_model is not None:
        model.load(load_saved_model)
    return model