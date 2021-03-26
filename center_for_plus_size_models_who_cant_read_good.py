import argparse
from fairseq.trainer import Trainer
from fairseq.models.roberta import RobertaModel
from fairseq.dataclass.configs import FairseqConfig
from common.task_utils import TASK_INFO
from download import get_dataset_path, get_model_path
from common import task_utils, argparsers
from roberta_custom import train
from os import path

# removes unnecessary stuff from a model, and saves it with "_slim" appended to filename
def main(args):
    if not path.isfile(args.filepath[0]):
        raise Exception(f'File {args.filepath[0]} does not exist')

    args = argparsers.args_finetune(namespace=args, parse_known=True)[0]
    print(args)
    task = args.task
    task_path = get_dataset_path(task)
    if args.arch == 'roberta_base':
        model_path = get_model_path('base') + '/model.pt'
    else:
        model_path = get_model_path('large') + '/model.pt'
    finetune_string = task_utils.get_finetune_string(task_path, model_path, args)
    trainer = train.get_trainer(finetune_string)
    print('got ze trainer')
    trainer.load_checkpoint(args.filepath[0])
    print('loaded checkpoint')
    # AssertionError: Optimizer does not match; please reset the optimizer (--reset-optimizer). FP16Optimizer vs FairseqAdam

if __name__ == "__main__":
    AP = argparse.ArgumentParser()
    AP.add_argument('filepath', type=str, nargs=1)
    AP.add_argument("--task", choices=TASK_INFO.keys(), required=True)
    ARGS = AP.parse_args()
    main(ARGS)