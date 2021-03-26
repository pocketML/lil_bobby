from glob import glob
from common import argparsers
from compression.distillation import data
from compression.distillation import data_augment
from analysis import parameters
import torch.nn as nn
import torch
from common.task_utils import TASK_INFO
from compression.distillation.models import (
    TangBILSTM,
    GlueBILSTM,
    BPE_FFN,
    DistLossFunction
)

# only works for single sentence prediction
def train_loop(model, criterion, optim, dl, device, num_epochs=10):
    for epoch in range(1, num_epochs + 1):
        print(f'* Epoch {epoch}')

        # train phase
        model.train()
        running_loss, running_corrects, num_examples = 0.0, 0.0, 0.0
        for x1, lens, target_labels, target_logits in dl['train']:
            x1 = x1.to(device)
            target_labels = target_labels.to(device)
            target_logits = target_logits.to(device)
            optim.zero_grad()
            torch.set_grad_enabled(True)
            out_logits = model(x1, lens)
            _, preds = torch.max(out_logits, 1)
            target_labels = target_labels.squeeze()
            loss = criterion(out_logits, target_logits, target_labels.squeeze())
            loss.backward()
            optim.step()
            running_loss += loss.item()
            running_corrects += torch.sum(preds == target_labels.data)
            num_examples += len(lens)
        print(f'|--> train loss: {running_loss / num_examples:.4f}')
        print(f'|--> train accuracy: {running_corrects / num_examples:.4f}')
        
        # validation phase
        model.eval()
        running_corrects, num_examples = 0.0, 0.0
        for x1, lens, target_labels, _ in dl['val']:
            x1 = x1.to(device)
            target_labels = target_labels.to(device)
            optim.zero_grad()
            torch.set_grad_enabled(False)
            out_logits = model(x1, lens)
            _, preds = torch.max(out_logits, 1)
            target_labels = target_labels.squeeze()
            running_corrects += torch.sum(preds == target_labels.data)
            num_examples += len(lens)
        print(f'|--> val accuracy: {running_corrects / num_examples:.4f}')

def main(args, sacred_experiment=None):
    print("Sit back, tighten your seat belt, and prepare for the ride of your life 🚀")

    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    use_gpu = not args.cpu
    epochs = args.epochs
    
    if args.generate_loss:
        data.generate_distillation_loss(args)
    elif args.augment:
        data_augment.augment(args.task, args.augment)
    elif args.play:
        torch.manual_seed(233)
        task = args.task
        student_type = args.student_arch

        if student_type == 'glue':
            model = GlueBILSTM(task, use_gpu)
        elif student_type == 'tang':
            model = TangBILSTM(task, use_gpu)
        elif student_type == 'wasserblat-ffn':
            model = BPE_FFN(task, use_gpu)

        if args.size:
            total_params, total_bits = parameters.get_model_size(model)
            print(type(model))
            print(f'total params: {total_params / 1000}K)')
            print(f'total size:   {total_bits / 8000000:.2f}MB')

        base_path = f'{TASK_INFO[task]["path"]}/distillation_data'
        distillation_data = []
        train_files = glob(f"{base_path}/*.tsv")
        for filename in train_files:
            loaded_data = data.load_distillation_data(filename)

            if distillation_data == []:
                distillation_data = loaded_data
            else:
                distillation_data[0].extend(loaded_data[0])
                distillation_data[1].extend(loaded_data[1])
                distillation_data[2].extend(loaded_data[2])
                if len(distillation_data) > 3:
                    distillation_data[3].extend(loaded_data[3])

        print(f"*** Loaded {len(distillation_data[0])} training data samples ***")

        val_data = data.load_val_data(task)
        model.to(device)

        print(f"*** Loaded {len(val_data[0])} validation data samples ***")

        criterion = DistLossFunction(0.5, nn.MSELoss(), nn.CrossEntropyLoss(), device)
        dataloaders = data.get_dataloader_dict(model, distillation_data, val_data)
        #optim = torch.optim.Adadelta(model.parameters())
        optim = model.get_optimizer()
        train_loop(model, criterion, optim, dataloaders, device, epochs)

if __name__ == "__main__":
    ARGS = argparsers.args_distill()
    main(ARGS)
