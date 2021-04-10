
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