# -*- coding: utf-8 -*-
"""
May 2021
Comment: This file uses Vader for sentiment analysis.
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


PATH = ""

dataset = pd.read_csv(PATH + "submissions.csv")
dataset["real_datetime"] = pd.to_datetime(dataset["created_utc"], unit='s').dt.floor('d')


"""VADER"""

filtered = dataset[dataset["real_datetime"] >= np.datetime64("2020-12-01")]
filtered = filtered[filtered["real_datetime"]  <= np.datetime64("2021-02-28")]
sample_df = filtered.groupby("real_datetime").sample(n=100, random_state=1)

analyser = SentimentIntensityAnalyzer()


def apply_vader(ds, col, sentiment):
    ds[sentiment] = ds[col].apply(lambda x: analyser.polarity_scores(str(x)))
    return ds

dataset = apply_vader(dataset.copy(), "title", "vader")


sample_df = pd.concat([sample_df, sample_df["vader"].apply(pd.Series)], axis =1)
sample_df = sample_df[["real_datetime", "neg", "neu", "pos", "compound"]]
sample_df["weighted_vader"] = -sample_df["neg"] + sample_df["pos"] 


sample_df.to_csv(PATH + "vader.csv")
