import argparse
from os.path import join as opj
import pandas as pd
from plot_topics import (
    extract_topics,
    save_topics,
    plot_topics,
    plot_model_eval,
    plot_lsi_coherence,
)
from topic import TopicModel


def label_dataset(df, ref_col, low_bound, high_bound, label_names):
    """
    Adds labels to the tweet dataframe

    Parameters
    ----------
    ref_col : str name of column in which we want to make the selection
    low_bound : list of str which set the lower boundary of choice
    high_bound : list of str which set the higher boundary of choice
    label_names : list of str that indicates the label names

    Returns
    -------

    df : pandas dataframe
    """
    assert len(low_bound) == len(
        high_bound
    ), "low_bound and high_bound must have equal size"

    assert len(low_bound) == len(
        label_names
    ), "low_bound and label_names must have equal size"
    label_col = []
    for n, label in enumerate(label_names):
        if n == len(label_names) - 1:
            label_set = df[
                (df[ref_col] >= low_bound[n]) & (df[ref_col] <= high_bound[n])
            ]
        else:
            label_set = df[
                (df[ref_col] >= low_bound[n]) & (df[ref_col] < high_bound[n])
            ]
        label_col.append([label] * len(label_set))
    df["label"] = [j for i in label_col for j in i]
    return df


def get_model(df, model_name):
    """
    gets the model with the highest coherence score
    for LDA or LSI
    """
    if model_name == "LDA":
        idx = df[df["coherence"] == df["coherence"].max()].index[0]
        return df["model"].loc[idx], df["alpha"].loc[idx], df["eta"].loc[idx]
    elif model_name == "LSI":
        idx = df[df["coherence"] == df["coherence"].max()].index[0]
        return df["model"].loc[idx]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--label", type=str, required=True)
    parser.add_argument("-a", "--analysis", nargs="+", type=str, required=True)
    args = parser.parse_args()

    # load the corpus of tweets
    data = pd.read_csv("~/00222_porcu/tweet_collection/clean_tweets.tsv", sep="\t")
    # add covid status labels to the corpus
    data = label_dataset(
        data,
        "date",
        [data["date"].loc[0], "2019-09-01", "2020-02-21"],
        ["2019-09-01", "2020-02-21", data["date"].loc[len(data) - 1]],
        ["no_covid", "pre-covid", "covid"],
    )

    if args.label == "all":
        print("Analysis all tweets:")
        print(f"Number of documents in the corpus: {len(data)}")

        # extract the tweets column from the dataframe
        CORPUS = data["text"].tolist()
    else:
        CORPUS = data[data["label"] == args.label]
        CORPUS = CORPUS["text"].tolist()
        print(f"Analysis {args.label} tweets:")
        print(f"Number of documents in the corpus: {len(CORPUS)}")

    TM = TopicModel(CORPUS)

    ensemble_workers = 6
    num_models = 12
    num_topics = 6
    distance_workers = 4
    passes = 10
    iterations = 10

    if "ensemble" in args.analysis:
        ENS_LDA = TM.ensembleLDA(
            num_topics, passes, num_models, ensemble_workers, distance_workers
        )

        DF_ENS = extract_topics(ENS_LDA.print_topics())
        save_topics(DF_ENS, f"Ensemble_LDA_topics_{args.label}_data")
        # DF_ENS = pd.read_csv(f"topic_data/Ensemble_LDA_topics_{args.label}_data.tsv", sep="\t")
        plot_topics(DF_ENS, f"ensemble_lda_topics_{args.label}_data")

    if "LDA" in args.analysis:

        ## Tune LDA

        LDA = TM.tune_LDA(
            iterations,
            passes,
            num_topics=[4, 6, 8, 10, 12],
            etas=["auto", "symmetric"],
            alphas=["auto", "symmetric", "asymmetric"],
        )
        DF_LDA = pd.DataFrame(LDA)
        save_topics(DF_LDA, f"lda_tuning_{args.label}")
        # DF_LDA = pd.read_csv(f"topic_data/lda_tuning_{args.label}.tsv", sep="\t")
        plot_model_eval(
            DF_LDA, "Alpha & Eta", "Coherence", f"Coherence_LDA_{args.label}_data"
        )
        model, alpha, eta = get_model(DF_LDA, "LDA")
        DF_SEL_LDA = extract_topics(model.print_topics())
        save_topics(DF_SEL_LDA, f"LDA_topics_{args.label}_data")
        # DF_SEL_LDA = pd.read_csv(f"topic_data/LDA_topics_{args.label}_data.tsv", sep="\t")
        plot_topics(DF_SEL_LDA, f"LDA_topics_{args.label}_data")

    if "LSI" in args.analysis:
        ## Tune LSI

        LSI = TM.tune_LSI([4, 6, 8, 10, 12])
        DF_LSI = pd.DataFrame(LSI)
        save_topics(DF_LSI, f"lsi_tuning_{args.label}")
        # DF_LSI = pd.read_csv(f"topic_data/lsi_tuning_{args.label}.tsv", sep="\t")
        plot_lsi_coherence(
            DF_LSI,
            "n_topics",
            "coherence",
            "N. topics",
            "Coherence",
            fig_name=f"LSI_coherence_{args.label}_data",
        )
        model = get_model(DF_LSI, "LSI")
        DF_SEL_LSI = extract_topics(model.print_topics())
        save_topics(DF_SEL_LSI, f"LSI_topics_{args.label}_data")
        # DF_SEL_LSI = pd.read_csv(f"topic_data/LSI_topics_{args.label}_data.tsv", sep="\t")
        plot_topics(DF_SEL_LSI, f"LSI_topics_{args.label}_data")
