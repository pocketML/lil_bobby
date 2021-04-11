from fairseq.data.data_utils import collate_tokens
from common import task_utils, data_utils
import os

def generate_distillation_loss(args, model):
    input_folder = args.generate_loss
    # We are working with regular pre-labelled data (not augmented data).
    augmented_data = input_folder != "processed"
    if augmented_data: # Load only sentences for augmented data.
        data = data_utils.load_augment_data(args.task, input_folder)
        sentence_pairs = len(data) == 2
    else: # Load sentences + target label.
        data = data_utils.load_train_data(args.task)
        sentence_pairs = len(data) == 3

    batch_size = 8
    n = len(data[0])
    output_path = f'{task_utils.TASK_INFO[args.task]["path"]}/distillation_data/'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_file = "train.tsv" if input_folder == "processed" else input_folder + ".tsv"
    label_fn = lambda label: model.task.label_dictionary.string([label + model.task.label_dictionary.nspecial])

    with open(output_path + output_file, 'w', encoding='utf-8') as out:
        for i in range(int((n - 1) / batch_size) + 1):
            start = i * batch_size
            end = start + batch_size if (start + batch_size) < n else n
            batch_sents = data[0][start : end]
            if not augmented_data: # Use target labels for regular labeled data.
                batch_targets = data[1][start : end]

            if sentence_pairs:
                batch_sents2 = data[-1][start : end]
                batch = collate_tokens(
                    [model.encode(sent1, sent2) for sent1, sent2 in zip(batch_sents, batch_sents2)],
                    pad_idx=1
                )
            else:
                batch = collate_tokens(
                    [model.encode(sent) for sent in batch_sents], 
                    pad_idx=1
                )
            batch_logits = model.predict('sentence_classification_head', batch, return_logits=True)
            if augmented_data: # Use predicted labels from logits for augmented data.
                batch_targets = [label_fn(label) for label in batch_logits.argmax(dim=1).tolist()]

            if sentence_pairs:
                for sent1, sent2, target, logits in zip(batch_sents, batch_sents2, batch_targets, batch_logits.tolist()):
                    logits_str = ','.join([str(x) for x in logits])
                    out.write(f'{sent1.strip()}\t{sent2.strip()}\t{target.strip()}\t{logits_str}\n')
            else:
                for sent, target, logits in zip(batch_sents, batch_targets, batch_logits.tolist()):
                    logits_str = ','.join([str(x) for x in logits])
                    out.write(f'{sent.strip()}\t{target.strip()}\t{logits_str}\n')
 