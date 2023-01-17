"""
Implements simple utilities for data selection, labeling, data interpolation
time series plotting and VAR and Granger causality pipeline from statsmodels
"""
from os.path import join as opj

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.api import VAR

def mult_plot(xdata, ydata, xlabel, ylabel, filename):
    """
    creates subplots of time series
    """

    fig, axs = plt.subplots(len(xdata))
    for n, (x, y) in enumerate(zip(xdata, ydata)):
        axs[n].plot(x, y)
        loc = plticker.MultipleLocator(base=25) 
        axs[n].xaxis.set_major_locator(loc)
        axs[n].set_xlabel(xlabel[n])
        axs[n].set_ylabel(ylabel[n])
    fig.tight_layout()
    plt.savefig(opj(f"time_series_results/{filename}.png"), dpi=100)
    #plt.show()

def plot_ts(data, x_data, y_data, title, xlabel, ylabel, line, save=False,
            fig_name=None, rot=0):
    """
    plot a time series
    """
    for series, x, y in zip(data, x_data, y_data):
        ax = sns.lineplot(x=x, y=y, data=series)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        plt.xticks(rotation=rot)
        if line:
            ax.axvline(line["point"], ls='--', color='r', linewidth=1)
            ax.text(0.5, 25, line["text"])
    plt.title(title)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"time_series_results/{fig_name}.png", dpi=100)

def plot_autocorr(data, lag):
    """
    plots for autocorrelation from statsmodels
    """
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data, lags=lag, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data, lags=lag, ax=ax2)
    plt.show()

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

    assert len(low_bound) == len(high_bound),\
    "low_bound and high_bound must have equal size"
    
    assert len(low_bound) == len(label_names),\
    "low_bound and label_names must have equal size"
    label_col = []
    for n, label in enumerate(label_names):
        if n == len(label_names)-1:
            label_set = df[(df[ref_col] >= low_bound[n]) & (df[ref_col] <= high_bound[n])]
        else:
            label_set = df[(df[ref_col] >= low_bound[n]) & (df[ref_col] < high_bound[n])]
        label_col.append([label]*len(label_set))
    df["label"] = [j for i in label_col for j in i]
    return df

def select_time_series(data, labels):
    """
    simple data selection from a pandas dataframe

    Parameters
    ----------

    data : pandas dataframe 

    labels : str name of the column to be used for the data selection
    
    Returns
    -------

    df : pandas dataframe of selected data
    """

    min_ts = data.groupby("label").size().min()
    time_series = []
    for label in labels:
        time_series.append(data[data["label"] == label].iloc[0:min_ts])
    df = pd.concat(time_series, axis=0).reset_index(drop=True)
    return df

def impute_time_series(data, col="date", method="zero_pad"):
    """
    Fills in missing days either by zeropadding
    or interpolating

    Parameters
    -----------
    df : pandas dataframe

    col : str name of column dataframe

    method : str if "zeropad", it fills in 
            zeros on the missing days
            if "interpolate", it interpolates
            using the 'nearest' method

    Returns
    -------

    df : pandas dataframe
    """

    df = data.copy(deep=True) # otherwhise it changes the original
    df[col] = pd.to_datetime(df[col])
    df.set_index(col, inplace=True)
    if method == "zero_pad":
        df = df.resample('D').sum().fillna(0)
    elif method == "interpolate":
        df = df.resample('D').sum().fillna(0)
        df = df.replace(0, np.NaN).interpolate(method='nearest')
    df.reset_index(level=0, inplace=True)
    return df

def is_stationary(data, method):
    """
    Test whether the time series is stationary
    Implements adfuller kpss form statsmodel

    Parameters
    ----------

    data : pandas dataframe of data

    method : str 'adfuller' or 'kpss'

    Returns:

    results : dict summary of of results of chosen method
              or None in case it was selected a wrong method name
    """

    if method == "kpss":
        return kpss(data)
    elif method == "adfuller":
        return adfuller(data)
    else:
        print(f"Method {method} is not implemented")
        return None

def cointegration(y, y0):
    """
    calls the cointegration method from statsmodels
    Parameters
    ----------

    y : numpy array or pandas dataframe/series

    y0 : numpy array or pandas dataframe/series

    Returns
    -------

    coint_stats : dict of statistical results
    """
    coint_stats = coint(y, y0, return_results=False)
    return coint_stats

class VarPipeline():
    """
    Wrapper around the VAR model and 
    Granger causality test from statsmodels

    Parameters
    ----------

    data : pandas dataframe with one time series pr column

    max_lag : int maximum number of lags to go through
    """
    def __init__(self, data, max_lag):
        self.data = data
        self.max_lag = max_lag
        self.model = VAR(self.data)

    def _get_lag(self):
        """
        select the best lag based on AIC
        """
        lags = self.model.select_order(self.max_lag)
        return lags.aic

    def fit_VAR(self):
        """
        fits the VAR model
        """
        model_fit = self.model.fit(self._get_lag())
        return model_fit

    def test_GC(self, caused, causing):
        """
        tests Granger-causality

        Parameters
        ----------

        caused : pandas series or numpy array

        causing : pandas series or numpy array
        
        Returns
        -------
        model : dict of statistical results 
        """
        model = self.fit_VAR().test_causality(caused, 
                                              causing, 
                                              kind='wald', 
                                              signif=0.05)
        return model

    def __str__(self):
        """ print lag chosen """
        return f"Lag output: {self._get_lag()}"
