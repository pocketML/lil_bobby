from common import argparsers, transponder
from compression.distillation import data
from compression.distillation import data_augment
from analysis import parameters
import torch.nn as nn
import torch
from compression.distillation.models import (
    TangBILSTM,
    GlueBILSTM,
    BPE_FFN,
    DistLossFunction
)

# only works for single sentence prediction
def train_loop(model, criterion, optim, dl, device, num_epochs=10, sacred_experiment=None):
    for epoch in range(1, num_epochs + 1):
        print(f'* Epoch {epoch}')

        for phase in ("train", "val"):
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss, running_corrects, num_examples = 0.0, 0.0, 0.0
            for x1, lens, target_labels, target_logits in dl[phase]:
                x1 = x1.to(device)
                target_labels = target_labels.to(device)
                if phase == "train":
                    target_logits = target_logits.to(device)
                optim.zero_grad()
                torch.set_grad_enabled(phase == "train")
                out_logits = model(x1, lens)
                _, preds = torch.max(out_logits, 1)
                target_labels = target_labels.squeeze()
                if phase == "train":
                    loss = criterion(out_logits, target_logits, target_labels.squeeze())
                    loss.backward()
                    optim.step()
                    running_loss += loss.item()
                running_corrects += torch.sum(preds == target_labels.data)
                num_examples += len(lens)
            accuracy = running_corrects / num_examples
            if phase == "train":
                print(f'|--> train loss: {running_loss / num_examples:.4f}')
            else:
                transponder.send_train_status(epoch, accuracy)
                if sacred_experiment is not None:
                    sacred_experiment.log_scalar("validation.acc", accuracy)
            print(f'|--> {phase} accuracy: {accuracy:.4f}')

def main(args, sacred_experiment=None):
    print("Sit back, tighten your seat belt, and prepare for the ride of your life 🚀")

    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    use_gpu = not args.cpu
    epochs = args.epochs

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    if args.generate_loss:
        data.generate_distillation_loss(args)
    elif args.augment:
        data_augment.augment(args.task, args.augment, args.seed)
    elif args.augment2:
        import compression.distillation.data_augment2 as data_augment2
        data_augment2.augment(args.task, args.augment2, args.seed)
    elif args.distillation:
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

        distillation_data = data.load_all_distillation_data(task)

        print(f"*** Loaded {len(distillation_data[0])} training data samples ***")

        val_data = data.load_val_data(task)
        model.to(device)

        print(f"*** Loaded {len(val_data[0])} validation data samples ***")

        criterion = DistLossFunction(0.5, nn.MSELoss(), nn.CrossEntropyLoss(), device)
        dataloaders = data.get_dataloader_dict(model, distillation_data, val_data)
        #optim = torch.optim.Adadelta(model.parameters())
        optim = model.get_optimizer()
        try:
            train_loop(
                model, criterion, optim, dataloaders, device,
                epochs, sacred_experiment=sacred_experiment
            )
        finally:
            model_name = (
                sacred_experiment.info["name"] if sacred_experiment is not None
                else args.student_arch
            )
            model.save(task, model_name)
            model_path = f"{model.get_model_path()}/{model_name}.pt"
            sacred_experiment.add_artifact(model_path)

if __name__ == "__main__":
    ARGS = argparsers.args_distill()
    main(ARGS)
