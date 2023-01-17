#!/usr/bin/env python3
"""
Simple twitter scraper that leverages on snscrape.
Notes: No need to filter retweets,
https://github.com/JustAnotherArchivist/snscrape/issues/163#issuecomment-737423269
Location based search is unreliable:
https://github.com/JustAnotherArchivist/snscrape/issues/145
It returned a very small number of tweets, removed it.
"""

from os.path import join as opj
import argparse
import pandas as pd
import snscrape.modules.twitter as sntwitter

def get_tweets(data_dict, keywords, start_date=None, stop_date=None, limit=5000):
    """
    Extracts tweets by using snscrape
    and returns a dictionary.
    """

    query = f"{keywords} since:{start_date} until:{stop_date} \
            lang:it filter:links filter:replies"
    all_tweets = sntwitter.TwitterSearchScraper(query).get_items()
    for n, tw in enumerate(all_tweets):
        if n > limit:
            return data_dict
        if tw.content not in data_dict["text"]:
            data_dict["text"].append(tw.content)
            data_dict["date"].append(tw.date)
            data_dict["keyword"].append(keywords)
            data_dict["user"].append(tw.username)
    return data_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keywords', nargs="+", type=str, required=True)
    parser.add_argument('-s', '--start_date', nargs="+", type=str, required=True)
    parser.add_argument('-e', '--end_date', nargs="+", type=str, required=True)
    parser.add_argument('-o', '--output_name', type=str, required=True)
    args = parser.parse_args()
    
    assert len(args.start_date) == len(args.end_date),\
    "start_date and end_date have different length"
    TW_DICT = {
        "text": [],
        "date": [],
        "keyword": [],
        "user": []
        }

    for key in args.keywords:
        for s, e in zip(args.start_date, args.end_date):
            TW_DICT = get_tweets(TW_DICT,
                                 key,
                                 start_date=s,
                                 stop_date=e)
    TF_DF = pd.DataFrame(TW_DICT)
    TF_DF.to_csv(opj("tweet_collection", f"{args.output_name}.tsv"), sep="\t", index=False)
    