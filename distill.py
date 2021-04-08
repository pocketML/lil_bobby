from common import argparsers, transponder, task_utils
from compression.distillation import data
from compression.distillation import data_augment
from analysis import parameters
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
        model_dir = task_utils.get_model_path(model.cfg["task"], "distilled")
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

def evaluate_distilled_model(model, dl, device, args, sacred_experiment=None):
    model.to(device)
    model.eval()
    running_corrects, num_examples = 0.0, 0.0
    iterator = tqdm(dl, leave=False) if args.loadbar else dl

    for x1, lens, target_labels, _ in iterator:
        x1 = x1.to(device)
        target_labels = target_labels.to(device)
        torch.set_grad_enabled(False)
        out_logits = model(x1, lens)
        _, preds = torch.max(out_logits, 1)
        target_labels = target_labels.squeeze()
        running_corrects += torch.sum(preds == target_labels.data)
        num_examples += len(lens)

    accuracy = running_corrects / num_examples
    if sacred_experiment is not None:
        sacred_experiment.log_scalar("validation.acc", accuracy)
    print(f'|--> val accuracy: {accuracy:.4f}')

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

    if args.generate_loss is not None:
        teacher_model = load_teacher(args.task, args.checkpoint_path, args.cpu)
        data.generate_distillation_loss(args, teacher_model)
    elif args.augment:
        data_augment.augment(args.task, args.augment, args.seed)

    elif args.eval is not None:
        model = load_student(task, student_type, use_gpu=use_gpu, model_name=args.eval)
        val_data = data.load_val_data(task)
        dl = data.get_dataloader_dict_val(model, val_data)
        evaluate_distilled_model(model, dl, device, args, sacred_experiment)

    elif args.distill:
        model = load_student(task, student_type, use_gpu=use_gpu)
        distillation_data = data.load_all_distillation_data(task)
        print(f"*** Loaded {len(distillation_data[0])} training data samples ***")

        val_data = data.load_val_data(task)
        model.to(device)
        print(f"*** Loaded {len(val_data[0])} validation data samples ***")

        criterion = DistLossFunction(
            args.alpha, 
            nn.MSELoss(), 
            nn.CrossEntropyLoss(), 
            temperature=temperature,
            device=device
        )
        dataloaders = data.get_dataloader_dict(model, distillation_data, val_data)
        print(f'*** Dataloaders created ***')

        optim = model.get_optimizer()
        train_loop(
            model, criterion, optim, dataloaders, device,
            args, epochs, sacred_experiment=sacred_experiment
        )

if __name__ == "__main__":
    ARGS = argparsers.args_distill()
    main(ARGS)
