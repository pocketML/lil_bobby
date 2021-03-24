from fairseq.models.roberta import RobertaModel
from common.task_utils import TASK_INFO
import download

from compression.distillation.student_models.tang_bilstm import TangBILSTM
from compression.distillation.student_models.glue_bilstm import (
    GlueBILSTM,
    get_loss_function as DistLossFunction
)
from compression.distillation.student_models.bpe_ffn import BPE_FFN

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
    model_dir = download.get_model_path('base' if arch == 'roberta_base' else 'large')
    model = RobertaModel.from_pretrained(model_dir, checkpoint_file='model.pt')
    if not use_cpu:
        model.cuda()
    model.eval()
    return model