from common import argparsers, transponder, model_utils, data_utils
from preprocessing import distillation_loss
import torch.nn as nn
import torch
from compression.distillation.models import DistLossFunction, load_student, load_teacher
from tqdm import tqdm

def save_checkpoint(model, student_arch, sacred_experiment=None):
    model_name = (
        sacred_experiment.info["name"] if sacred_experiment is not None
        else student_arch
    )
    model.save(model_name)
    if sacred_experiment:
        model_dir = model_utils.get_model_path(model.cfg["task"], "distilled")
        model_path = f'{model_dir}/{model_name}.pt'
        sacred_experiment.add_artifact(model_path)

# only works for single sentence prediction
def train_loop(model, criterion, optim, dl, device, args, num_epochs, sacred_experiment=None):
    best_val_acc = 0
    no_improvement = 0
    for epoch in range(1, num_epochs + 1):
        print(f'* Epoch {epoch}')

        for phase in ("train", "val"):
            if phase == "train":
                model.train()
            else:
                model.eval()

            iterator = tqdm(dl[phase], leave=False) if args.loadbar else dl[phase]

            running_loss, running_corrects, num_examples = 0.0, 0.0, 0.0
            for x1, lens, target_labels, target_logits in iterator:
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
                if accuracy > best_val_acc:
                    print(f'Saving new best model')
                    best_val_acc = accuracy
                    save_checkpoint(model, args.student_arch, sacred_experiment)
                    no_improvement = 0
                else:
                    no_improvement += 1
                transponder.send_train_status(epoch, accuracy.item())
                if sacred_experiment is not None:
                    sacred_experiment.log_scalar("validation.acc", accuracy.item())
            print(f'|--> {phase} accuracy: {accuracy:.4f}')
            if no_improvement == args.early_stopping:
                break

def main(args, sacred_experiment=None):
    print("Sit back, tighten your seat belt, and prepare for the ride of your life 🚀")

    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    use_gpu = not args.cpu
    epochs = args.epochs
    task = args.task
    student_type = args.student_arch
    temperature = args.temperature
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    elif args.distill:
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
    ARGS = argparsers.args_distill()
    main(ARGS)
