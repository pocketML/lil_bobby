from sklearn.metrics import f1_score
import random

def precision(tp, fp):
    return tp / (tp + fp)

def recall(tp, fn):
    return tp / (tp + fn)

def sklearn_f1(tp, fn, fp):
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2 * ((p * r) / (p + r))

def our_f1(tp, fn, fp):
    return tp / (tp + 0.5 * (fp + fn))

def get_stuff(preds, targets):
    tp = sum([int(p == 1 and t == 1) for p, t in zip(preds, targets)])
    fp = sum([int(p == 1 and t == 0) for p, t in zip(preds, targets)])
    fn = sum([int(p == 0 and t == 1) for p, t in zip(preds, targets)])
    return tp, fp, fn

preds = [random.randint(0,1) for _ in range(1000)]
targets = [random.randint(0,1) for _ in range(1000)]
tp, fp, fn = get_stuff(preds, targets)

print(tp, fp, fn)
print(f'sklearn: {sklearn_f1(tp, fn, fp)}')
print(f'our own: {our_f1(tp, fn, fp)}')
print(f'real sklearn: {f1_score(preds, targets)}')