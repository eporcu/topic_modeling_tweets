"""
Implements a wrapper for gensim ensembleLDA, LSI, LDA
the coherence evaluation method
"""


import gensim
from gensim import corpora
from gensim.models import (
    LdaMulticore,
    EnsembleLda,
    LdaModel,
    LsiModel,
    CoherenceModel,
    TfidfModel,
)
import numpy as np
import pandas as pd
import spacy


class Preproc:
    """
    Performs simple gensim preprocessing,
    lemmatization and transformations, dictionary,
    bag of words, tfidf.

    Parameters
    ----------
    corpus : list or array of documents

    below : int  keep token less than no_below
            documents (absolute number)

    above : float keep tokens more than
            no_above documents (fraction of total corpus size)
    """

    def __init__(self, corpus, below=20, above=0.5):
        self.corpus = corpus
        self.below = below
        self.above = above

    def preprocessing(self):
        """
        simple wrapper for the gensim preprocessing
        plus lemmatization.

        Returns
        -------
        lemmatized_corpus : list of lemmatized documents
        """

        corpus = [
            gensim.utils.simple_preprocess(doc, deacc=True, min_len=2, max_len=15)
            for doc in self.corpus
        ]
        nlp = spacy.load("it_core_news_sm")
        lemmatized_corpus = []
        for sent in corpus:
            doc = nlp(" ".join(sent))
            lemmatized_corpus.append([word.lemma_ for word in doc])
        return lemmatized_corpus

    def to_dictionary(self, corpus):
        """
        turns the corpus into a dictionary
        (applies a simple filter)

        Parameters
        ----------

        corpus : list of lemmatized documents

        Returns
        -------

        corpus_dict : dict of corpus
        """
        corpus_dict = corpora.Dictionary(corpus)
        corpus_dict.filter_extremes(no_below=self.below, no_above=self.above)
        return corpus_dict

    def to_bow(self, corpus, dictionary):
        """
        takes a dictionary and returns a bag of words
        """

        bow = [dictionary.doc2bow(doc) for doc in corpus]
        return bow

    def to_Tfidf(self, bow):
        """
        applies tfidf tranformation on the bag of words
        """
        tfidf = TfidfModel(bow)
        return tfidf[bow]


class TopicModel(Preproc):
    """
    Wrapper around three gensim topic modelling algorythms
    EnsembleLDA, LDA and LSI.
    Inherits preprocessing and transformation steps from
    the class Preproc.

    Parameters
    ----------

    corpus : list or array of documents

    below : int  keep token less than no_below
            documents (absolute number)

    above : float keep tokens more than
            no_above documents (fraction of total corpus size)

    """

    def __init__(self, corpus, below=20, above=0.5):
        super().__init__(corpus, below, above)
        self.corpus = self.preprocessing()
        self.dictionary = self.to_dictionary(self.corpus)
        self.bow = self.to_bow(self.corpus, self.dictionary)
        self.tfidf = self.to_Tfidf(self.bow)

    def ensembleLDA(
        self, num_topics, passes, num_models, ensemble_workers, distance_workers
    ):
        """
        Wrapper of ensemble LDA

        Parameters
        ----------

        num_topics : int number of topics requested
        passes : int number of passes through the corpus
        num_models : int number of models trained by each processes
        ensemble_workers : int number of processes
        distance_workers : int distance computations

        Returns
        -------

        ensemble : object ensemble lda output

        """
        dictionary = self.dictionary
        temp = dictionary[0]  # it allows to get the dictionary.

        ensemble = EnsembleLda(
            corpus=self.bow,
            id2word=dictionary.id2token,
            num_topics=num_topics,
            passes=passes,
            num_models=num_models,
            topic_model_class=LdaMulticore,
            ensemble_workers=ensemble_workers,
            distance_workers=distance_workers,
            random_state=1,
        )

        return ensemble

    def LDA(
        self, iterations, num_topics, passes, eval_every=None, alpha="auto", eta="auto"
    ):
        """
        Parameters
        ----------

        iterations : int  Maximum number of iterations through the corpus

        num_topics : int number of topics requested

        passes : int number of passes through the corpus

        eval_every : int Log perplexity is estimated every that many updates

        alpha : float, numpy.ndarray of float, list of float, str
                a-priori belief on document-topic distribution

        eta : float, numpy.ndarray of float, list of float, str
              a-priori belief on topic-word distribution

        Returns
        -------

        lda : object lad object output

        """

        dictionary = self.dictionary
        temp = dictionary[0]  # it allows to get the dictionary.
        lda = LdaModel(
            corpus=self.bow,
            id2word=dictionary.id2token,
            chunksize=len(self.corpus),
            alpha="auto",
            eta="auto",
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every,
            per_word_topics=True,
            random_state=1,
        )
        return lda

    def tune_LDA(self, iterations, passes, num_topics, alphas, etas, eval_every=None):
        """
        Parameters
        ----------

        iterations : int  Maximum number of iterations through the corpus

        passes : int number of passes through the corpus
        num_topics : int number of topics requested
        alphas : list of float, numpy.ndarray of float, list of str
                a-priori belief on document-topic distribution
        etas : list of float, numpy.ndarray of float, list of str
              a-priori belief on topic-word distribution
        eval_every : int Log perplexity is estimated every that many updates

        Returns
        -------

        res_dict : dict of results with keys
                    "model" list of object models
                    "n_topic" number of topics requested
                    "coherence" coherence score
                    "alpha" alpha value
                    "eta" eta value

        """

        dictionary = self.dictionary
        temp = dictionary[0]  # it allows to get the dictionary.
        res_dict = {"model": [], "n_topic": [], "coherence": [], "alpha": [], "eta": []}
        for n in num_topics:
            for eta in etas:
                for alpha in alphas:
                    lda = LdaModel(
                        corpus=self.bow,
                        id2word=dictionary.id2token,
                        chunksize=len(self.corpus),
                        alpha=alpha,
                        eta=eta,
                        iterations=iterations,
                        num_topics=n,
                        passes=passes,
                        eval_every=eval_every,
                        per_word_topics=True,
                        random_state=1,
                    )

                    coherence = CoherenceModel(
                        model=lda,
                        texts=self.corpus,
                        dictionary=dictionary,
                        coherence="c_v",
                    )
                    res_dict["model"].append(lda)
                    res_dict["coherence"].append(coherence.get_coherence())
                    res_dict["n_topic"].append(n)
                    res_dict["eta"].append(eta)
                    res_dict["alpha"].append(alpha)
        return res_dict

    def LSI(self, num_topics):

        """
        Wrapper for LSI method,

        Parameters
        ----------

        num_topics : int number of topics requested

        Returns
        -------

        lsi_model : object LSI output
        """

        corpus_tfidf = self.tfidf
        dictionary = self.dictionary
        temp = dictionary[0]
        lsi_model = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
        return lsi_model

    def tune_LSI(self, num_topics):
        """
        method for simple lsi tuning,

        Parameters
        ----------

        num_topics : int number of topics requested

        Returns
        -------

        res_dict : dict of results with keys
                    "n_topics" number of topics requested
                    "model" list of object models
                    "coherence" coherence score
        """
        corpus_tfidf = self.tfidf
        dictionary = self.dictionary
        temp = dictionary[0]
        res_dict = {"coherence": [], "model": [], "n_topics": []}
        for n in num_topics:
            lsi_model = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n)
            coher = CoherenceModel(
                model=lsi_model,
                texts=self.corpus,
                dictionary=dictionary,
                coherence="c_v",
            )
            res_dict["coherence"].append(coher.get_coherence())
            res_dict["model"].append(lsi_model)
            res_dict["n_topics"].append(n)
        return res_dict
