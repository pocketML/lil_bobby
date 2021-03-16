from fairseq.models.roberta import RobertaModel
from common.task_utils import TASK_INFO

from compression.distillation.student_models.tang_bilstm import (
    TangBILSTM,
    get_loss_function as TangLoss
)

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