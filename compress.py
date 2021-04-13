import torch
import torch.nn as nn
from common import argparsers, data_utils, task_utils
from compression.distill import train_loop
from compression.distillation.models import DistLossFunction, load_student
import evaluate
from compression.quantization import post_training as ptq
from analysis import parameters

def main(args, sacred_experiment=None):
    print("Sit back, tighten your seat belt, and prepare for the ride of your life 🚀")

    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    use_gpu = not args.cpu
    task = args.task
    student_type = args.student_arch
    seed = args.seed
    if args.named_seed is not None:
        seed = task_utils.SEED_DICT[args.named_seed]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if "quantize" in args.compression_actions:
        model_name = "tang_hadfield_alpha_0_3"
        model = load_student(task, student_type, use_gpu=use_gpu, model_name=model_name)
        model.to(device)
        model.cfg['use-gpu'] = use_gpu
        val_data = data_utils.load_val_data(task)
        dl = data_utils.get_dataloader_dict_val(model, val_data)

        backend = 'fbgemm'
        torch.backends.quantized.engine = backend
        import warnings
        warnings.simplefilter("ignore", UserWarning)
        for emb in [None, 'static']:
            for lstm in [None, 'dynamic']:
                for cls in [None, 'static', 'dynamic']:
                    print(f'emb: {emb}, lstm: {lstm}, class: {cls}')
                    current = model
                    if emb == 'static':
                        current = ptq.quantize_embeddings(current, args)
                    if lstm == 'dynamic':
                        current = ptq.quantize_encoder(current)
                    if cls ==  'static':
                        current = ptq.quantize_classifier(current, args, type='static')
                    elif cls == 'dynamic':
                        current = ptq.quantize_classifier(current, args, type='dynamic')
                    evaluate.evaluate_distilled_model(current, dl, device, args, None)
                    parameters.print_model_disk_size(current)
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
            temperature=temperature,
            device=device
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
