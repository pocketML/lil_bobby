from functools import reduce
import matplotlib.pyplot as plt
import numpy as np

from common import model_utils

# from https://github.com/OrdnanceSurvey/GeoDataViz-Toolkit/tree/master/Colours
COLORS_QUALITATIVE = [
	"#FF1F5B",
	"#00CD6C",
	"#009ADE",
	"#AF58BA",
	"#FFC61E",
	"#F28522",
	"#A0B1BA",
	"#A6761D",
	"#E9002D",
	"#FFAA00",
	"#00B000"]

COLORS_Q_CBLIND_FRIENDLY = [
    COLORS_QUALITATIVE[1],
    COLORS_QUALITATIVE[2],
    COLORS_QUALITATIVE[3],
    COLORS_QUALITATIVE[4]
]

def create_show_pie_chart(data, labels):
    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw=dict(aspect="equal"))
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40, colors=COLORS_Q_CBLIND_FRIENDLY)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
            bbox=bbox_props, zorder=0, va="center")
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)
    plt.tight_layout()
    plt.savefig('sample.pdf', bbox_inches="tight")
    plt.show()

def weight_pie_chart(model, arch):
    grouped = model_utils.group_params_by_layer(model, arch)
    if arch in model_utils.MODEL_INFO.keys(): # either roberta large or base
        p_count = [0,0,0]
        labels = ['Sentence encoder', 'Transformer layers', 'LM Head']
        for key in grouped:
            for _,params in grouped[key]:
                nparams = reduce(lambda acc, x: acc * x, params.shape, 1)
                if 'encoder' in key:
                    p_count[0] += nparams
                elif 'layer_' in key:
                    p_count[1] += nparams
                elif 'head' in key:
                    p_count[2] += nparams

        total = sum(p_count)
        pcts = [f'{(c / total) * 100.0:.2f}%\n({"{:,}".format(c).replace(",", " ")})' for c in p_count]
        labels = [labels[i] + '\n' + pcts[i] for i in range(len(labels))]
        create_show_pie_chart(p_count, labels)
    if arch == 'glue':
        # full GloVe 42B token, 22m vocab, 300 dim embedding param count = 22000000 * 300 = 6 600 000 000
        # original ELMO embedding param count = 93 600 000
        p_count = [93600000, 0, 0] # first entry is GloVe 42B token, 22m vocab, 300 dim embedding
        labels = ['Embedding layer', 'BiLSTM', 'Classifier']
        grouped = model_utils.group_params_by_layer(model, arch)
        for _, params in grouped['bilstm']:
            p_count[1] += reduce(lambda acc, x: acc * x, params.shape, 1)
        for _, params in grouped['classifier']:
            p_count[2] += reduce(lambda acc, x: acc * x, params.shape, 1)
        total = sum(p_count)
        pcts = [f'{(c / total) * 100.0:.2f}%\n({"{:,}".format(c).replace(",", " ")})' for c in p_count]
        labels = [labels[i] + '\n' + pcts[i] for i in range(len(labels))]
        create_show_pie_chart(p_count, labels)        


# TODO: only works for RoBERTa models at the moment
def weight_histogram_for_all_transformers(model, arch, num_bins=2000):
    layers = model_utils.group_params_by_layer(model, arch)
    transformers = [layer for layer in layers.keys() if 'layer_' in layer]
    n = len(transformers)
    ncols = 4
    nrows = int(n / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18,15))
    for i, ax in enumerate(axs.flat):
        name = transformers[i]
        layer = layers[name]
        weights = concat_weights_in_layer(layer)
        ax.hist(weights, bins=num_bins)
        ax.set(xlabel='Weight value',ylabel='Frequency', xlim=(-.3,.3), ylim=(0,74000),title=name)
    for ax in axs.flat:
        ax.label_outer()
    plt.show()

def weight_histogram_for_layer(layer, num_bins=1000):
    weights = concat_weights_in_layer(layer)
    plt.hist(weights, bins=num_bins)
    plt.show()