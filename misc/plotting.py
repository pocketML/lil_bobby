import matplotlib.pyplot as plt
import argparse
font = {'size': 16}
plt.rc('font', **font)
import load_results

# Data is task -> list of [model_name, acc_1, acc_2, size]
MODEL_DATA = {
    "sst-2": [
        ("RoBERTa Large", 96.56, 1426.02 * 1000),
        ("GLUE Baseline", 90.2, 681.128 * 1000),
        ("TinyBERT", 93.0, 116 * 1000)
    ],
    "qqp": [
        ("RoBERTa Large", 92.15, 1426.02 * 1000),
        ("GLUE Baseline", 85.7, 681.128 * 1000),
        ("TinyBERT", 90.4, 116 * 1000)
    ],
    "mnli": [
        ("RoBERTa Large", 90.15, 1426.02 * 1000),
        ("GLUE Baseline", 73.15, 681.128 * 1000),
        ("TinyBERT", 84.5, 116 * 1000)
    ]
}

# 90.3275
# 89.975

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
        (-20, -26), (-60, 12), (-100, -7), (-30, -26), (10, -7), (-30, 12),
        (-10, -26), (-25, 10), (-30, 12), (-65, 12), (-70, 12), (-84, 12)
    ],
    "qqp": [
        (-35, -24), (-40, -24), (12, -6), (-25, -26), (12, -7), (-25, -25),
        (-25, 10), (-66, -7), (-26, -26), (-100, -2), (-70, 12), (-84, 12)
    ],
    "mnli": [
        (12, -6), (10, -14), (-102, -4), (10, -6), (-28, 10), (10, -7),
        (16, -10), (-68, -7), (-30, -26), (-100, -2), (-70, 12), (-155, 4)
    ]
}

def load_model_data():
    for model_name, model_list in MODELS_TO_LOAD:
        for task_specific_model in  model_list:
            results = load_results.get_results_for_distilled_model(task_specific_model, "prune_quant")
            data = load_results.get_experiment_data(results, "final")

            task = data["task"]
            acc = data["acc"][0]
            size = data["theoretical_size"]

            MODEL_DATA[task].append((model_name, acc * 100, size * 1000))

def get_annotation_position(x, y, model_index, task, axis):
    offset_x, offset_y = TEXT_OFFSETS[task][model_index]

    x_transformed, y_transformed = axis.transData.transform((x, y))

    new_x, new_y = axis.transData.inverted().transform((x_transformed + offset_x, y_transformed + offset_y))

    return new_x, new_y

def get_pareto_data(sorted_data, staircase=False):
    points_x = []
    points_y = []
    if staircase:
        min_y = min(x[1] for x in sorted_data)
        points_y.append(min_y)

    highest_acc = 0
    models_on_skyline = set()
    for model_name, accuracy, size in sorted_data:
        y = highest_acc
        if accuracy > highest_acc:
            highest_acc = accuracy
            y = accuracy
            models_on_skyline.add(model_name)
            if not staircase:
                points_x.append(size)
                points_y.append(y)

        if staircase:
            points_x.extend([size] * 2)
            points_y.extend([y] * 2)

    if staircase:
        points_x.append(points_x[-1])

    return points_x, points_y, models_on_skyline

def plot_pareto(data, pareto_x, pareto_y, task, skyline_models):
    fig, ax = plt.subplots()

    points_x = [x[2] for x in data]
    points_y = [x[1] for x in data]
    skyline_x = [x[2] for x in data if x[0] in skyline_models]
    skyline_y = [x[1] for x in data if x[0] in skyline_models]

    width = 12 #3.487
    height = (width * 0.5625)
    fig.set_size_inches(width, height, forward=True)

    ax.set_xscale("log")
    ax.set_ylim(min(points_y) - 3, max(points_y) + 3)
    x_lim_min = 8 if task == "sst-2" else 15
    ax.set_xlim(x_lim_min, 3_500_000)

    ax.grid(b=True, which="major", axis="both", linestyle="--")

    ax.set_xlabel("Size (KB)")
    ax.set_ylabel("Accuracy (%)")

    for index, (model_name, p_y, p_x) in enumerate(data):
        model_index = MODEL_NAMES.index(model_name) if model_name in MODEL_NAMES else index
        ax.annotate(
            model_name, get_annotation_position(p_x, p_y, model_index, task, ax), fontsize=14
        )

    ax.plot(pareto_x, pareto_y, linewidth=2, c="#2EC038")

    radius = 50
    border_width = 30
    ax.scatter(skyline_x, skyline_y, s=radius + border_width, color="red", zorder=2.5)
    ax.scatter(points_x, points_y, s=radius, color="#0D4B89", zorder=3)

    filename = f"misc/pareto_{task}.pdf"

    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    AP = argparse.ArgumentParser()

    AP.add_argument("task", choices=("sst-2", "qqp", "mnli"))
    AP.add_argument("--staircase", action="store_true")

    ARGS = AP.parse_args()

    load_model_data()
    sorted_by_size = sorted(MODEL_DATA[ARGS.task], key=lambda x: x[2])
    x_data, y_data, models_on_skyline = get_pareto_data(sorted_by_size, ARGS.staircase)
    plot_pareto(sorted_by_size, x_data, y_data, ARGS.task, models_on_skyline)
