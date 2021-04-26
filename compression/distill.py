import torch
from tqdm import tqdm
from common import transponder, model_utils
from compression import prune

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

            running_loss, running_corrects, num_examples = 0.0, 0.0, 0
            for x1, lens, target_labels, target_logits in iterator:
                if task_utils.is_sentence_pair(model.cfg['task']):
                    x1 = x1[0].to(device), x1[1].to(device)
                else:
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
                    loss = criterion(out_logits, target_logits, target_labels)
                    loss.backward()
                    optim.step()
                    running_loss += loss.item() * len(lens)
                running_corrects += torch.sum(preds == target_labels.data).item()
                num_examples += len(lens)

            accuracy = 0 if num_examples == 0 else running_corrects / num_examples
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
                transponder.send_train_status(epoch, accuracy)
                if sacred_experiment is not None:
                    sacred_experiment.log_scalar("validation.acc", accuracy)
            print(f'|--> {phase} accuracy: {accuracy:.4f}')
            if no_improvement == args.early_stopping:
                return
