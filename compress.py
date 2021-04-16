import torch
import torch.nn as nn
from common import argparsers, data_utils, task_utils
from compression.distill import train_loop
from compression.distillation.models import DistLossFunction, load_student
import evaluate
from compression.quantization import post_training as ptq
from analysis import parameters
import warnings

warnings.simplefilter("ignore", UserWarning)

def main(args, sacred_experiment=None):
    print("Sit back, tighten your seat belt, and prepare for the ride of your life 🚀")

    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    use_gpu = not args.cpu
    task = args.task
    student_type = args.student_arch
    seed = args.seed
    if args.seed_name is not None:
        seed = task_utils.SEED_DICT[args.seed_name]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if "quantize" in args.compression_actions:
        model_name = args.load_trained_model
        model = load_student(task, student_type, use_gpu=use_gpu, model_name=model_name)
        model.to(device)
        dl = data_utils.get_dataloader_dict_val(model, data_utils.load_val_data(task))
        backend = 'fbgemm'
        torch.backends.quantized.engine = backend

        if args.ptq_embedding:
            print("** quantizing embedding layer **")
            model = ptq.quantize_embeddings(model, args)
        if args.dq_encoder:
            print("** quantizing encoder **")
            model = ptq.quantize_encoder(model)
        if args.dq_classifier:
            print("** quantizing classifier **")
            model = ptq.quantize_classifier(model, args, type='dynamic')
        elif args.ptq_classifier:
            print("** quantizing classifier **")
            model = ptq.quantize_classifier(model, args, type='static')

        evaluate.evaluate_distilled_model(model, dl, device, args, None)
        parameters.print_model_disk_size(model)
        print()

    if "distill" in args.compression_actions:
        epochs = args.epochs
        temperature = args.temperature

        model = load_student(task, student_type, use_gpu=use_gpu)
        distillation_data = data_utils.load_all_distillation_data(task)
        print(f"*** Loaded {len(distillation_data[0])} training data samples ***")
        val_data = data_utils.load_val_data(task)
        model.to(device)
        print(f"*** Loaded {len(val_data[0])} validation data samples ***")

        criterion = DistLossFunction(
            args.alpha, 
            nn.MSELoss(), 
            nn.CrossEntropyLoss(), 
            device,
            temperature=temperature,
        )

        dataloaders = data_utils.get_dataloader_dict(model, distillation_data, val_data)
        print(f'*** Dataloaders created ***')

        optim = model.get_optimizer()
        train_loop(
            model, criterion, optim, dataloaders, device,
            args, epochs, sacred_experiment=sacred_experiment
        )

if __name__ == "__main__":
    ARGS = argparsers.args_compress()[0]

    main(ARGS)
