import matplotlib.pyplot as plt
import argparse
font = {'size': 14}
plt.rc('font', **font)
import load_results

# Data is task -> list of [model_name, acc_1, acc_2, size]
MODEL_DATA = {
    "sst-2": [
        ("RoBERTa Large", 96.53, 1426.02 * 1000),
        ("GLUE Baseline", 90.2, 681.128 * 1000),
        ("TinyBERT", 93.0, 116 * 1000)
    ],
    "qqp": [
        ("RoBERTa Large", 92.15, 1426.02 * 1000),
        ("GLUE Baseline", 85.7, 681.128 * 1000),
        ("TinyBERT", 91.1, 116 * 1000)
    ],
    "mnli": [
        ("RoBERTa Large", 90.92, 1426.02 * 1000),
        ("GLUE Baseline", 73.15, 681.128 * 1000),
        ("TinyBERT", 84.5, 116 * 1000)
    ]
}

LETTERS = ("A", "B", "C")

MODEL_NAMES = (
    [f"BiLSTM {letter}" for letter in LETTERS] +
    [f"RNN {letter}" for letter in LETTERS] +
    [f"FFN {letter}" for letter in LETTERS]
)

MODELS_TO_LOAD = []
for model_name, model_group in zip(MODEL_NAMES, load_results.EXTRA_COMPRESSION_MODELS):
    models = []
    for task_model in model_group:
        models.append(task_model)
    MODELS_TO_LOAD.append((model_name, models))

TEXT_OFFSETS = {
    "sst-2": [
        (-25, 12), (-18, 12), (-20, 12), (-35, 12), (-45, 12), (-60, -24), (-100, 12),
        (-25, 12), (-18, 12), (-20, 12), (-35, 12), (-45, 12), (-60, -24), (-100, 12)
    ],
    "qqp": [
        (-25, 12), (-18, 12), (-20, 12), (-35, 12), (-45, 12), (-60, -24), (-100, 12),
        (-10, -25), (-65, -6), (-25, 10), (-35, -24), (-90, -6), (-60, 12), (-90, 12)
    ],
    "mnli": [
        (-15, 10), (-25, 10), (-25, 10), (-35, 10), (-90, -6), (-60, 12), (-135, -4),
        (-15, 10), (-25, 10), (-25, 10), (-35, 10), (-90, -6), (-60, 12), (-135, -4)
    ]
}

def load_model_data():
    for model_name, model_list in MODELS_TO_LOAD:
        for task_specific_model in  model_list:
            results = load_results.get_results_for_distilled_model(task_specific_model, "prune_quant")
            data = load_results.get_experiment_data(results, "final")

            task = data["task"]
            acc, acc_2 = data["acc"]
            size = data["theoretical_size"]

            if task == "mnli":
                acc = (acc + acc_2) / 2

            MODEL_DATA[task].append((model_name, acc * 100, size * 1000))

def get_annotation_position(x, y, model_index, task, axis):
    offset_x, offset_y = TEXT_OFFSETS[task][model_index]

    x_transformed, y_transformed = axis.transData.transform((x, y))

    new_x, new_y = axis.transData.inverted().transform((x_transformed + offset_x, y_transformed + offset_y))

    return new_x, new_y

def get_pareto_data(sorted_data):
    points_x = []
    min_y = min(x[1] for x in sorted_data)
    points_y = [min_y]
    highest_acc = 0
    models_on_skyline = set()
    for model_name, accuracy, size in sorted_data:
        y = highest_acc
        if accuracy > highest_acc:
            highest_acc = accuracy
            y = accuracy
            models_on_skyline.add(model_name)
        points_x.extend([size] * 2)
        points_y.extend([y] * 2)

    points_x.append(points_x[-1])

    return points_x, points_y

def plot_pareto(data, pareto_x, pareto_y, task):
    fig, ax = plt.subplots()

    width = 6.5 #3.487
    height = (width / 1.2)
    fig.set_size_inches(width, height, forward=True)

    ax.set_xscale("log")
    ax.set_ylim(min(pareto_y) - 3, max(pareto_y) + 3)
    x_lim_min = 10 if task == "sst-2" else 15
    ax.set_xlim(x_lim_min, 3_500_000)

    ax.set_xlabel("Size (KB)")
    ax.set_ylabel("Accuracy (%)")

    points_x = [x[2] for x in data]
    points_y = [x[1] for x in data]

    for index, (model_name, p_y, p_x) in enumerate(data):
        ax.annotate(
            model_name, get_annotation_position(p_x, p_y, index, task, ax), fontsize=12
        )

    ax.plot(pareto_x, pareto_y, linestyle=":", linewidth=2, c="red")

    ax.scatter(points_x, points_y, s=50)

    filename = f"pareto_{task}.png"

    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    AP = argparse.ArgumentParser()

    AP.add_argument("task", choices=("sst-2", "qqp", "mnli"))

    ARGS = AP.parse_args()

    load_model_data()
    sorted_by_size = sorted(MODEL_DATA[ARGS.task], key=lambda x: x[2])
    x_data, y_data = get_pareto_data(sorted_by_size)
    plot_pareto(sorted_by_size, x_data, y_data, ARGS.task)
