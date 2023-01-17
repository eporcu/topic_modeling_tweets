from os.path import join as opj
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_topics(topics, filename):
    """
    saves topics df as tsv
    """
    topics.to_csv(opj("topic_data", f"{filename}.tsv"), sep="\t", index=False)


def extract_topics(topics):
    """
    extracts topics from the output of
    gensim print_topics() method.
    """
    keyword, weights, n_topic = [], [], []
    for topic in topics:
        top = re.sub("[^a-z0-9.]", " ", topic[1]).split()
        topic_weights = top[0::2]
        weights.append([float(i) for i in topic_weights])
        keyword.append(top[1::2])
        n_topic.append([topic[0] + 1] * len(topic_weights))
    topic_dict = {
        "keyword": [i for k in keyword for i in k],
        "weight": [i for w in weights for i in w],
        "Ntopic": [i for t in n_topic for i in t],
    }
    df = pd.DataFrame(topic_dict)
    return df


def plot_topics(topics, filename):
    """
    plots all topics and words with the relative weights
    axes rotation trick for seaborn FacetGrid got it from:
    https://stackoverflow.com/questions/60077401/rotate-x-axis-labels-facetgrid-seaborn-not-working
    """
    ax = sns.FacetGrid(
        topics,
        col="Ntopic",
        sharex=False,
        col_wrap=len(topics["Ntopic"].unique()) // 2,
        height=4,
        aspect=0.65,
    )
    ax.map_dataframe(sns.barplot, x="keyword", y="weight", color="red", saturation=0.1)
    ax.set_axis_labels(x_var="Words", y_var="Weights")
    for n, axes in enumerate(ax.axes.flat):
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
        axes.set_title(f"Topic {n+1}")
    plt.tight_layout()
    plt.savefig(opj("topic_data", f"{filename}.png"), dpi=100)
    # plt.show()


def plot_model_eval(data, xlabel, ylabel, fig_name=None):
    """
    plot model evaluation: coherence
    """
    data["alpha_eta"] = data[["alpha", "eta"]].agg("_".join, axis=1)
    grid = sns.FacetGrid(data, col="n_topic", palette="tab20c", col_wrap=5, sharey=True)
    grid.map(plt.plot, "alpha_eta", "coherence", marker="o")
    grid.set(xlabel=xlabel, ylabel=ylabel)
    titles = data["n_topic"].unique()
    for n, axes in enumerate(grid.axes.flat):
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
        axes.set_title(f"Topic {titles[n]}")
    plt.tight_layout()
    # plt.show()
    plt.savefig(opj("topic_data", f"{fig_name}.png"), dpi=100)


def plot_lsi_coherence(
    data, x_data, y_data, xlabel, ylabel, fig_name=None, title="LSI Coherence"
):
    """
    plot LSI coherence
    """
    ax = sns.lineplot(x=x_data, y=y_data, data=data, marker="o")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.title(title)
    # plt.show()
    plt.savefig(opj("topic_data", f"{fig_name}.png"), dpi=100)
