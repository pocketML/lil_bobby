import task_difficulty
from common import data_utils, task_utils

def load_validation_sentences(task):
    val_data = data_utils.load_train_data(task, ds_type="dev")
    if task_utils.is_sentence_pair(task):
        sents_1, _, sents_2 = val_data
        sentences = [sents_1, sents_2]
    else:
        sents_1, _ = val_data
        sentences = [sents_1]
    sentences = [list(reversed([x.strip() for x in sents])) for sents in sentences]
    return sentences

def find_word_probabilities(sent_predictions):
    word_preds = {x: {} for x in task_difficulty.MODEL_ARCHS}
    word_probs = {x: {} for x in task_difficulty.MODEL_ARCHS}
    for arch in task_difficulty.MODEL_ARCHS:
        # 0 is the index for corrects, 1 for wrongs
        for correct_idx in (0,1):
            for sent in sent_predictions[arch][correct_idx]:
                for word in " ".joint(sent).split(" "):
                    if word_preds[arch].get(word) is None:
                        word_preds[arch][word] = [0,0]
                    word_preds[arch][word][correct_idx] += 1

        for word, counts in word_preds[arch].items():
            sum_occ = sum(counts)
            word_probs[arch][word] = (counts[0] / sum_occ, sum_occ)
    return word_probs

def sentencen_probs_unseen_words(task):
    pass

def print_word_probabilites(task):
    val_sents = load_validation_sentences(task)
    predictions = task_difficulty.load_predicted_sentences(task, val_sents)
    word_probs = find_word_probabilities(predictions)

    train_og_words = task_difficulty.get_word_occurences(task)
    train_all_words = task_difficulty.get_word_occurences(task, include_augmented=True)

    print(f'Task: {task}')
    print(f'Total unique words in train set: {len(train_og_words)}')
    print(f'Total unique words in train + augmented set: {len(train_all_words)}\n')

    for arch in task_difficulty.MODEL_ARCHS:
        print(f'************* {arch} *************')
        print(f'Overall accuracy: {len(predictions[arch][0]) / (len(predictions[arch][0]) + len(predictions[arch][1])):.4f}')
        more_than_20 = list(filter(lambda x: x[1][1] >= 20, word_probs[arch].items()))
        
        print('Worst accuracy on words with more than 20 occurences')
        more_than_20.sort(key=lambda x: x[1][0])
        for word, prob in more_than_20[:20]:
            print(f'{word:<16} acc: {prob[0]:.3f}{"":<10} count: {prob[1]}')
        print()

        print('Best accuracy on words with more than 20 occurences')
        more_than_20.sort(key=lambda x: -x[1][0])
        for word, prob in more_than_20[:20]:
            print(f'{word:<16} acc: {prob[0]:.3f}{"":<10} count: {prob[1]}')
        print()

def unseen_words_in_val(task):
    val_sents = load_validation_sentences(task)
    sent_predictions = task_difficulty.load_predicted_sentences(task, val_sents)
    #word_probs = find_word_probabilities(predictions)
    train_og_words = task_difficulty.get_word_occurences(task)
    train_all_words = task_difficulty.get_word_occurences(task, include_augmented=True)

    seen_in_og = []
    seen_in_aug = []
    unseen = []
    total_words = 0

    for sent in val_sents:
        for word in " ".join(sent).split(" "):
            if word in train_og_words:
                seen_in_og.append(word)
            elif word in train_all_words:
                seen_in_aug.append(word)
            else:
                unseen.append(word)
            total_words += 1
    
    seen_in_og_set = set(seen_in_og)
    seen_in_aug_set = set(seen_in_aug)
    unseen_set = set(unseen)

    print(f'Task: {task}')
    print(f'Total words in val set: {total_words}')
    print()
    print(f'Val words also in train set:      {len(seen_in_og)} ({(len(seen_in_og)/total_words)*100:.2f}%)')
    print(f'Val words only in augmented set:  {len(seen_in_aug)} ({(len(seen_in_aug)/total_words)*100:.2f}%)')
    print(f'Val words unseen during training: {len(unseen)} ({(len(unseen)/total_words)*100:.2f}%)')
    print()
    print(f'Unique words in train set: {len(train_og_words)}')
    print(f'Unique words in train + augmented set: {len(train_all_words)}')
    print()
    print(f'Unique val words also in train set:      {len(seen_in_og_set)}')
    print(f'Unique val words only in augmented set:  {len(seen_in_aug_set)}')
    print(f'Unique val words unseen during training: {len(unseen_set)}')
    print()

    for arch in task_difficulty.MODEL_ARCHS:
        print(f'************* {arch} *************')
        print(f'Overall accuracy: {len(sent_predictions[arch][0]) / (len(sent_predictions[arch][0]) + len(sent_predictions[arch][1])):.4f}')
        
        # 0 is the index for corrects, 1 for wrongs
        only_og_preds = [0,0]
        one_augment_or_more = [0,0]
        two_augment_or_more = [0,0]
        one_unseen_or_more = [0,0]
        two_unseen_or_more = [0,0]

        for correct_idx in (0,1):
            for sent in sent_predictions[arch][correct_idx]:
                num_augment = [w for w in " ".join(sent).split(" ") if w in seen_in_aug_set]
                num_unseen = [w for w in " ".join(sent).split(" ") if w in unseen_set]
                if len(num_augment) > 0:
                    one_augment_or_more[correct_idx] += 1
                if len(num_augment) > 1:
                    two_augment_or_more[correct_idx] += 1
                if len(num_unseen) > 0:
                    one_unseen_or_more[correct_idx] += 1
                if len(num_unseen) > 1:
                    two_unseen_or_more[correct_idx] += 1
                if len(num_unseen) == 0 and len(num_augment) == 0:
                    only_og_preds[correct_idx] += 1
        
        print(f'Acc for sents with only words from train set:           {only_og_preds[0] / sum(only_og_preds):.4f} ({only_og_preds[0]}/{sum(only_og_preds)})')
        print(f'Acc for sents with one or more words unique to aug set: {one_augment_or_more[0] / sum(one_augment_or_more):.4f} ({one_augment_or_more[0]}/{sum(one_augment_or_more)})')
        print(f'Acc for sents with two or more words unique to aug set: {two_augment_or_more[0] / sum(two_augment_or_more):.4f} ({two_augment_or_more[0]}/{sum(two_augment_or_more)})')
        print(f'Acc for sents with one or more unseen words:            {one_unseen_or_more[0] / sum(one_unseen_or_more):.4f} ({one_unseen_or_more[0]}/{sum(one_unseen_or_more)})')
        print(f'Acc for sents with two or more unseen words:            {two_unseen_or_more[0] / sum(two_unseen_or_more):.4f} ({two_unseen_or_more[0]}/{sum(two_unseen_or_more)})')
        print()

if __name__ == "__main__":
    #print_word_probabilites("sst-2")
    unseen_words_in_val('sst-2')