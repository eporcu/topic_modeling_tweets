"""
summary plots to visualize results of scraped tweets
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_time_series(data, x_data, y_data, tick_dens,
                     title, xlabel, ylabel, save=False,
                     fig_name=None, rot=45):
    """
    plot a time series
    """
    ax = sns.lineplot(x=x_data, y=y_data, data=data)
    ax.set_xticks(np.arange(0, len(data), tick_dens))
    ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.xticks(rotation=rot)
    plt.title(title)
    plt.show()
    if save:
        plt.savefig(f"{fig_name}.png", dpi=100)

def plot_tweet_count(data, x_data, y_data, xlabel, ylabel,
                     title, save=False, fig_name=None):
    """
    barplot with for tweet counts for each keyword
    """
    ax = sns.barplot(data, x=x_data, y=y_data)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.xticks(rotation=25)
    plt.title = title
    plt.show()
    if save:
        plt.savefig(f"{fig_name}.png", dpi=100)