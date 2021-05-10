# -*- coding: utf-8 -*-
"""
May 2021
Comment: This file uses FinBERT for sentiment analysis.
"""


import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import numpy as np

PATH = ""

dataset = pd.read_csv(PATH + "submissions.csv")
dataset["real_datetime"] = pd.to_datetime(dataset["created_utc"], unit='s').dt.floor('d')

"""FINBERT"""
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

texte= "GME stock is rising quickly"
print(classifier(texte)[0])

filtered = dataset[dataset["real_datetime"] >= np.datetime64("2020-12-01")]
filtered = filtered[filtered["real_datetime"]  <= np.datetime64("2021-02-28")]
sample_df = filtered.groupby("real_datetime").sample(n=100, random_state=1)

def apply_finbert(ds, col, sentiment):
    ds[sentiment] = ds[col].apply(lambda x: classifier(str(x))[0])
    return ds

start_time = time.time()
sample_df = apply_finbert(sample_df.copy(), "title", "finbert")
print(time.time() - start_time, "seconds")

sample_df = pd.concat([sample_df, sample_df["finbert"].apply(pd.Series)], axis=1)

sample_df.to_csv(PATH + "finbert.csv")
