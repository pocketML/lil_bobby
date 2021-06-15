import torch
import torch.nn as nn

from tqdm import tqdm

from common import task_utils, data_utils, transponder

def save_checkpoint(model, student_arch, sacred_experiment=None):
    model_name = (
        sacred_experiment.info["name"] if sacred_experiment is not None
        else student_arch
    )
    model.save(model_name)

def train_loop(model, criterion, optim, dl, device, args):
    val_accuracy = 0
    non_embedding_params = model.non_embedding_params()

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
                examples = len(lens[0])
            else:
                x1 = x1.to(device)
                examples = len(lens)
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
                nn.utils.clip_grad_norm_(non_embedding_params, model.cfg['clip-grad'])
                optim.step()
                running_loss += loss.item() * examples

            running_corrects += torch.sum(preds == target_labels.data).item()
            num_examples += examples

        accuracy = 0 if num_examples == 0 else running_corrects / num_examples
        if phase == "train":
            print(f'|--> train loss: {running_loss / num_examples:.4f}')
        else:
            val_accuracy = accuracy
        print(f'|--> {phase} accuracy: {accuracy:.4f}')

    return val_accuracy


def distill_model(task, model, device, args, callback, sacred_experiment):
    model.to(device)
    val_data = data_utils.load_val_data(task)
    print(f"*** Loaded {len(val_data[0])} validation data samples ***")

    train_data = data_utils.load_all_distillation_data(
        task, 
        only_original_data=args.only_original_data,
        bootstrap_data_ratio=args.bootstrap_data_ratio,
        downsample_distill_data=args.downsample_data
    )
    print(f"*** Loaded {len(train_data[0])} training data samples ***")

    print('*** Preparing data for Dataloaders ***')
    dataloader_train = data_utils.get_train_dataloader(model, train_data, loadbar=args.loadbar)
    dataloader_val = data_utils.get_val_dataloader(model, val_data, loadbar=args.loadbar)
    dataloaders = {"train": dataloader_train, "val": dataloader_val}
    print('*** Dataloaders created ***')

    criterion = model.get_combined_loss_function(
        args.alpha, 
        nn.MSELoss(), 
        nn.CrossEntropyLoss(), 
        device,
        temperature=args.temperature,
    )
    optim = model.get_optimizer()

    best_val_acc, no_improvement = 0, 0,
    for epoch in range(1, args.epochs + 1):
        print(f'* Epoch {epoch}')

        val_acc = train_loop(model, criterion, optim, dataloaders, device, args)

        if val_acc > best_val_acc:
            print(f'Saving new best model')
            best_val_acc = val_acc
            save_checkpoint(model, args.student_arch, sacred_experiment)
            no_improvement = 0
        else:
            no_improvement += 1

            transponder.send_train_status(epoch, val_acc)
            if sacred_experiment is not None:
                sacred_experiment.log_scalar("validation.acc", val_acc)

            if callback is not None:
                model = callback(model, args, epoch)
            if no_improvement == args.early_stopping:
                break