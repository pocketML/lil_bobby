import load_results

# Data is task -> list of [model_name, acc_1, acc_2, size]
MODEL_DATA = {
    "sst-2": [
        ("RoBERTa Large", 96.53, None, 1426.02)
    ],
    "qqp": [
        ("RoBERTa Large", 90.35, 90.06, 1426.02)
    ],
    "mnli": [
        ("RoBERTa Large", 92.20, 89.63, 1426.02)
    ]
}

# Ordered by largest to smallest.
MODELS_TO_LOAD = [
    ("BiLSTM B", load_results.EXTRA_COMPRESSION_MODELS[1]), # BiLSTM (large)
    ("RNN C", load_results.EXTRA_COMPRESSION_MODELS[5]), # RNN (pretty small)
    ("FFN C", load_results.EXTRA_COMPRESSION_MODELS[8]) # FFN (L1 cache)
]

def load_model_data():
    for model_name, model_list in MODELS_TO_LOAD:
        for task_specific_model in  model_list:
            results = load_results.get_results_for_distilled_model(task_specific_model, "prune_quant")
            data = load_results.get_experiment_data(results)

            task = data["task"]
            acc_1, acc_2 = data["acc"]
            size = data["theoretical_size"]

            MODEL_DATA[task] = (model_name, acc_1, acc_2, size)

def plot_stuff():
    pass

if __name__ == "__main__":
    load_model_data()
