# -*- coding: utf-8 -*-
"""
April 2021
Comment: This file read the submission dataset and an external dataset queried with the Reddit API. 
We can quickly show that Pushift's submissions are incomplete with the real number of comments higher on some days with the Reddit API.
Also, this code shows that the external submissions dataset is far from being complete and cannot be used.
"""

import pandas as pd
import nltk

nltk.download('stopwords')
nltk.download('punkt')

PATH = ""

"""##################"""
"""IMPORTING DATASETS"""
"""##################"""
dataset = pd.read_csv(PATH + "submissions.csv")
external_ds = pd.read_csv(PATH + "reddit_wsb.csv")

"""#####################"""
"""TESTING THE KAGGLE DS"""
"""#####################"""
count_dates = dataset["real_datetime"].value_counts().rename_axis('date').reset_index(name='count')
missing_days = pd.date_range(start = '2020-01-01', end = '2021-03-31' ).difference(dataset.real_datetime)
count_dates = pd.concat([count_dates, pd.DataFrame(missing_days, columns=["date"])])


external_ds["real_datetime"] = pd.to_datetime(external_ds["created"], unit='s').dt.floor('d')
count_dates2 = external_ds["real_datetime"].value_counts().rename_axis('date').reset_index(name='count')
missing_days = pd.date_range(start = '2020-01-01', end = '2021-03-31' ).difference(external_ds.real_datetime)
count_dates2 = pd.concat([count_dates2, pd.DataFrame(missing_days, columns=["date"])])

#we can show that the Kaggle dataset is pretty bad in terms of frequency and that we have missing submissions on some days with Puhsift.
comparing = count_dates.merge(count_dates2, how='left', on='date')
