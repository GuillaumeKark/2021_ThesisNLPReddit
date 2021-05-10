# -*- coding: utf-8 -*-
"""
April 2021
Comment: This file read the submission dataset and creates the wordclouds in the Master Thesis.
We first preprocess the text and the apply the Counter() function to retrieve the most popular words.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import emoji
import twokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')

PATH = ""
dataset = pd.read_csv(PATH + "submissions.csv")

"""#######################"""
"""WORDCLOUD PREPROCESSING"""
"""#######################"""

#After viewing the wordclouds, two additional operations could be performed : removed "'s" and numbers.
#This is trivial to add with quick functions and has been done in more advanced chapters of this thesis.

def lower_case(ds, col):
    ds[col] = ds[col].apply(lambda x: str(x).lower())
    return ds

dataset = lower_case(dataset.copy(), "title") #short


# emoji
def is_emoji(s):
    return s in emoji.UNICODE_EMOJI["en"].keys()

# add space near your emoji
def add_space(text):
    return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()


def emoji_spacer(ds, col):
    ds[col] = ds[col].apply(lambda x: add_space(str(x)))
    return ds

dataset = emoji_spacer(dataset.copy(), "title")


#removing punctuations
import string
def remove_punctuation(ds, col):
    ds[col] = ds[col].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation + "“”")))
    return ds

dataset = remove_punctuation(dataset.copy(), "title")


def tokenize_without_stopwords(ds, col):
    stop_words_nltk = set(nltk.corpus.stopwords.words('english'))   
    func = (lambda x: [i for i in twokenize.tokenizeRawTweetText(x) if not i in stop_words_nltk])
    ds[col] = ds[col].apply(func)
    return ds

dataset = tokenize_without_stopwords(dataset.copy(), "title")


"""#################"""
"""WORDCLOUD BUILDER"""
"""#################"""

from collections import Counter
from wordcloud import WordCloud


#all dataset
counter = Counter()
top100 = dataset['title'].apply(lambda x: counter.update(x))
top100 = dict(counter.most_common(100))
my_wordcloud = WordCloud(background_color="white", width=1600, height=800, font_path='fonts/seguiemj.ttf').generate_from_frequencies(top100)
plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("All submissions", fontsize=60)
plt.imshow(my_wordcloud)


#november2020
filtered = dataset[dataset["real_datetime"]  >= np.datetime64("2020-11-01")]
filtered = filtered[filtered["real_datetime"]  <= np.datetime64("2020-11-30")]

counter = Counter()
top100 = filtered['title'].apply(lambda x: counter.update(x))
top100 = dict(counter.most_common(100))
my_wordcloud = WordCloud(background_color="white", width=1600, height=800, font_path='fonts/seguiemj.ttf').generate_from_frequencies(top100)
plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Submissions on November 2020", fontsize=60)
plt.imshow(my_wordcloud)


#december2020
filtered = dataset[dataset["real_datetime"]  >= np.datetime64("2020-12-01")]
filtered = filtered[filtered["real_datetime"]  <= np.datetime64("2020-12-31")]

counter = Counter()
top100 = filtered['title'].apply(lambda x: counter.update(x))
top100 = dict(counter.most_common(100))
my_wordcloud = WordCloud(background_color="white", width=1600, height=800, font_path='fonts/seguiemj.ttf').generate_from_frequencies(top100)
plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Submissions on December 2020", fontsize=60)
plt.imshow(my_wordcloud)


#january2021
filtered = dataset[dataset["real_datetime"]  >= np.datetime64("2021-01-01")]
filtered = filtered[filtered["real_datetime"]  <= np.datetime64("2021-01-31")]

counter = Counter()
top100 = filtered['title'].apply(lambda x: counter.update(x))
top100 = dict(counter.most_common(100))
my_wordcloud = WordCloud(background_color="white", width=1600, height=800, font_path='fonts/seguiemj.ttf').generate_from_frequencies(top100)
plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Submissions on January 2021", fontsize=60)
plt.imshow(my_wordcloud)

#feburary2021
filtered = dataset[dataset["real_datetime"]  >= np.datetime64("2021-02-01")]
filtered = filtered[filtered["real_datetime"]  <= np.datetime64("2021-02-28")]

counter = Counter()
top100 = filtered['title'].apply(lambda x: counter.update(x))
top100 = dict(counter.most_common(100))
my_wordcloud = WordCloud(background_color="white", width=1600, height=800, font_path='fonts/seguiemj.ttf').generate_from_frequencies(top100)
plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Submissions on February 2021", fontsize=60)
plt.imshow(my_wordcloud)
