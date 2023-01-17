"""
"""
from os.path import join as opj

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
import time_series_analysis as ts



if __name__ == '__main__':

    TS_DF = pd.read_csv(opj("tweet_collection", "clean_tweets.tsv"), sep="\t")
    ## get the count of tweets per day
    TS_COUNT = TS_DF.groupby(["date"], as_index=False).count()
    # with interpolation
    TS_INT = ts.impute_time_series(TS_COUNT, "date", "interpolate")

    # reintroduce labels and rename "text" as "counts" and get rid of "keyword"
    TS_INT = ts.label_dataset(TS_INT,
                              "date",
                              [TS_INT["date"].loc[0], "2019-09-01", "2020-02-21"], 
                              ["2019-09-01", "2020-02-21", TS_INT["date"].loc[len(TS_INT)-1]],
                              ["no_covid", "pre-covid", "covid"])
    TS_INT = TS_INT.rename(columns={"text":"count"}).drop("keyword", axis=1)
    # save dataframe as tsv
    TS_INT.to_csv("time_series_results/Tweet_count.tsv", "\t", index=False)

    # select only pre-covid and covid of same size

    PRE_POST = ts.select_time_series(TS_INT, ["pre-covid", "covid"])
    
    line_dict={"point":18313, "text":"first case"}
    ts.plot_ts([PRE_POST], ["date"], ["count"], "Pre-covid and covid periods", "Date", "Tweets/day", save=True,
               fig_name="Pre-covid and covid period", line=line_dict)
   
    ## Granger Causality

    GRANG_DF = ts.select_time_series(TS_INT, ["pre-covid", "covid"])
    # divide the two time series
    pre_covid = GRANG_DF["count"].loc[GRANG_DF["label"]=="pre-covid"].values
    covid = GRANG_DF["count"].loc[GRANG_DF["label"]=="covid"].values
    GR_DF = pd.DataFrame({"pre-covid": pre_covid, "covid": covid})
    
    # Assess stationarity
    ADF = ts.is_stationary(GR_DF["pre-covid"], method="adfuller")
    KPSS = ts.is_stationary(GR_DF["pre-covid"], method="kpss")
    print(f"pre-covid stationarity - adfuller: p = {ADF[1]}")
    print(f"pre-covid stationarity - KPSS: p = {KPSS[1]}")

    ADF = ts.is_stationary(GR_DF["covid"], method="adfuller")
    KPSS = ts.is_stationary(GR_DF["covid"], method="kpss")
    print(f"covid stationarity - adfuller: p = {ADF[1]}")
    print(f"covid stationarity - KPSS: p = {KPSS[1]}")

    # Cointegration test (check reltionship between time series)

    print("Cointegration test:\n",\
        ts.cointegration(GR_DF["pre-covid"], GR_DF["covid"]))

    # Take first order difference

    GR_DF = GR_DF.diff().dropna()


    # check again stationarity

    ADF = ts.is_stationary(GR_DF["pre-covid"], method="adfuller")
    KPSS = ts.is_stationary(GR_DF["pre-covid"], method="kpss")
    print(f"pre-covid stationarity - adfuller: p = {ADF[1]}")
    print(f"pre-covid stationarity - KPSS: p = {KPSS[1]}")

    ADF = ts.is_stationary(GR_DF["covid"], method="adfuller")
    KPSS = ts.is_stationary(GR_DF["covid"], method="kpss")
    print(f"covid stationarity - adfuller: p = {ADF[1]}")
    print(f"covid stationarity - KPSS: p = {KPSS[1]}")

    # Apply VAR model and Granger-causality
    MAX_LAG = 15 # arbitrary maximum lag
    VAR = ts.VarPipeline(GR_DF, MAX_LAG)
    print(VAR)
    RES = VAR.fit_VAR()
    #RES.plot_acorr()
    #plt.show()
    print(VAR.test_GC("covid", "pre-covid"))
    print(VAR.test_GC("pre-covid", "covid"))


    ## Granger causality tweet - real covid data

    C19 = pd.read_csv("dpc-covid19-ita-andamento-nazionale.csv")
    # select data of the first wave
    C19["data"] = [data.split("T")[0] for data in C19["data"]] # drop time in the "data" column
    C19 = C19[C19["data"] <= '2020-06-01'].reset_index(drop=True)

    # select twitter data
    TWEET = TS_INT[(TS_INT["date"] >= "2020-02-24") & (TS_INT["date"] <= "2020-06-01")].reset_index(drop=True)

    assert len(TWEET) == len(C19), f"TWEET and C19 have different size {len(TWEET)}, {len(C19)}"

    TW_CV19 = pd.DataFrame({"tweets":TWEET["count"].values, 
                            "positives":C19["nuovi_positivi"].values})

    ts.mult_plot(xdata=[TWEET["date"].values, C19["data"].values],
                 ydata=[TWEET["count"].values, C19["nuovi_positivi"].values],
                 xlabel=["Dates", "Dates"], ylabel=["N. tweet/day", "new positives/day"],
                 filename="tweet_covid_cases")

    # Cointegration test (check reltionship between time series)
    print("Cointegration test:\n",\
        ts.cointegration(TW_CV19["tweets"], TW_CV19["positives"]))

    # Assess stationarity
    ADF = ts.is_stationary(TW_CV19["tweets"], method="adfuller")
    KPSS = ts.is_stationary(TW_CV19["tweets"], method="kpss")
    print(f"tweets stationarity - adfuller: p = {ADF[1]}")
    print(f"tweets stationarity - KPSS: p = {KPSS[1]}")

    ADF = ts.is_stationary(TW_CV19["positives"], method="adfuller")
    KPSS = ts.is_stationary(TW_CV19["positives"], method="kpss")
    print(f"covid stationarity - adfuller: p = {ADF[1]}")
    print(f"covid stationarity - KPSS: p = {KPSS[1]}")

    TW_CV19 = TW_CV19.diff().diff().dropna()

    ADF = ts.is_stationary(TW_CV19["tweets"], method="adfuller")
    KPSS = ts.is_stationary(TW_CV19["tweets"], method="kpss")
    print(f"tweets stationarity - adfuller: p = {ADF[1]}")
    print(f"tweets stationarity - KPSS: p = {KPSS[1]}")

    print("Run stationarity tests again")
    ADF = ts.is_stationary(TW_CV19["positives"], method="adfuller")
    KPSS = ts.is_stationary(TW_CV19["positives"], method="kpss")
    print(f"covid stationarity - adfuller: p = {ADF[1]}")
    print(f"covid stationarity - KPSS: p = {KPSS[1]}")

    # Apply VAR model and Granger-causality
    print("Run GRanger-causality")
    MAX_LAG = 15 
    VAR = ts.VarPipeline(TW_CV19, MAX_LAG)
    RES = VAR.fit_VAR()
    print(VAR)
    #RES.plot_acorr()
    #plt.show()
    print(VAR.test_GC("positives", "tweets"))
    print(VAR.test_GC("tweets", "positives"))
