from os.path import join as opj
import pandas as pd
from summary_plots import plot_time_series
from tw_cleaner import Cleaner, TwSanityCheck

if __name__ == '__main__':

    DATA = pd.read_csv(opj("tweet_collection", "tweets.tsv"), sep="\t", index_col=False)
    TWEET_CLEANER = Cleaner()
    TWEETS = [TWEET_CLEANER(tweet) for tweet in DATA["text"]]
    DATES = [date.split()[0] for date in DATA["date"]] # drop irrelevant time
    DF = pd.DataFrame({"text":TWEETS, "date":DATES, "keyword":DATA["keyword"]})
    SC = TwSanityCheck()
    DF = SC(DF).sort_values(by="date").reset_index(drop=True)
    print(DF)
    DF.to_csv(opj("tweet_collection", "clean_tweets.tsv"), sep="\t", index=False)
