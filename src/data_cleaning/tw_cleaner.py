"""
simple classes to clean up tweets
"""
from os.path import join as opj
import json
import re

from nltk.corpus import stopwords

import pandas as pd

class Cleaner():
    """
    Cleaning class:
    """

    def __init__(self, pattern=None, language="italian"):
        self.pattern = pattern
        self.language = language

    def clean_text(self, text):
        """
        simple implementation of regex to clean tweets,
        it gives the chance to provide a non default pattern.
        regex pattern:
        ((^| ).(( ).)*( |$)) removes recurring
        single characters
        (@\w+) removes reference to twitter users
        e.g. @username
        (#\w+) removes hashtags
        (\W)|(_) removes all alphanumeric characters
        punctuation and underscores
        (https\S+) removes urls
        [0-9\n] removes numbers
        """
        if self.pattern is not None:
            pattern = self.pattern
        else:
            pattern = "([0-9\n])|((^| ).(( ).)*( |$))|(@\w+)|(#\w+)|(\W)|(_)|(https\S+)"
        return re.sub(pattern, " ", text).lower().split()

    def rm_stop_words(self, text):
        """
        removes stopwords by leveraging on
        ntlk stopwords module, joins elements
        of the list.
        The stop words extension "stopwords-it.json"
        has been taken from the following project:
        https://github.com/stopwords-iso/stopwords-it
        """
        with open("stopwords-it.json") as extended_sw:
            sw = json.load(extended_sw)
        sw_list = stopwords.words(self.language)
        sw_list.extend(sw)
        return ' '.join([item for item in text
                         if item not in sw_list])

    def __call__(self, text):
        """
        makes the class callable
        """
        return self.rm_stop_words(self.clean_text(text))

class TwSanityCheck():
    def __init__(self, text_col_name="text", kw_col_name="keyword", language="italian"):
        
        self.text_col_name = text_col_name
        self.kw_col_name = kw_col_name
        self.language = language

    def drop_retweets(self, data):
        """
        removes duplicates from the dataframe
        """
        idx = data[data.duplicated(subset=self.text_col_name)].index
        data = data.drop(idx).reset_index(drop=True)
        return data

    def drop_no_keywords_tweet(self, data):
        """
        If the text does not contain the keywords
        tweet is droped from the dataframe.
        """
        idx2drop = []
        for n, key in enumerate(data[self.kw_col_name]):
            if key not in data[self.text_col_name].loc[n]:
                idx = data[data[self.text_col_name] == data[self.text_col_name].loc[n]].index
                idx2drop.append(idx[0])
        data = data.drop(idx2drop).reset_index(drop=True)
        return data

    def drop_single_words_numbers(self, data, n_words=2):
        """
        """
        idx2drop = []
        for n, doc in enumerate(data[self.text_col_name]):
            if type(doc) is not str or len(doc.split()) <= n_words:
                idx = data[data[self.text_col_name] == data[self.text_col_name].loc[n]].index 
                idx2drop.append(idx[0])
        data = data.drop(idx2drop).reset_index(drop=True)
        return data

    def __call__(self, data):

        df = self.drop_retweets(data)
        df = self.drop_no_keywords_tweet(df)
        df = self.drop_single_words_numbers(df)
        return df
        
