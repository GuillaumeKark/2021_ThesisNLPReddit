# -*- coding: utf-8 -*-
"""
May 2021
Comment: This file creates the Buy/Hold/Sell indicator.
"""

import pandas as pd
import numpy as np
import emoji
import twokenize
import nltk
import spacy
nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')


PATH = ""

dataset = pd.read_csv(PATH + "submissions.csv")
dataset["real_datetime"] = pd.to_datetime(dataset["created_utc"], unit='s').dt.floor('d')

"""##############"""
"""PRE-PROCESSING"""
"""##############"""

def remove_apos_s(ds, col):
    ds[col] = ds[col].apply(lambda x: str(x).replace("'s", ""))
    ds[col] = ds[col].apply(lambda x: str(x).replace("’s", ""))
    return ds


def lower_case(ds, col):
    ds[col] = ds[col].apply(lambda x: str(x).lower())
    return ds


# emoji
def is_emoji(s):
    return s in emoji.UNICODE_EMOJI["en"].keys()

# add space near your emoji
def add_space(text):
    return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()


def emoji_spacer(ds, col):
    ds[col] = ds[col].apply(lambda x: add_space(str(x)))
    return ds


def remove_trailwhitespace(ds, col):
    ds[col] = ds[col].apply(lambda x: ' '.join([token for token in x.split()]))
    return ds

dataset = remove_apos_s(dataset.copy(), "title") #short=
dataset = lower_case(dataset.copy(), "title") #short
dataset = emoji_spacer(dataset.copy(), "title")
dataset = remove_trailwhitespace(dataset.copy(), "title")


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


hold_voc = ["hold", "holding", "held", "holds", "diamonds", "diamond", "hand", "hands", "still", '💎']
sell_voc = ["sell", "sold", "selling", "resell", "sale", "sells"]
buy_voc = ["buy", "purchase", "buying", "purchased", "purchasing", "bought", "purchases", "buys", "moon", "rocket", "mars", '🚀']



def create_flag(ds, col, lis, flag_name):
    ds[flag_name] = ds[col].apply(lambda x: (set(lis) & set(x)) != set())
    return ds

dataset = create_flag(dataset.copy(), "title", hold_voc, "hold")
dataset = create_flag(dataset.copy(), "title", sell_voc, "sell")
dataset = create_flag(dataset.copy(), "title", buy_voc, "buy")

#other category tested ("flag squeeze")
def flag_sq(ds, col):
    sq1 = ["short", "squeeze"]
    ds["squeeze"] = ds[col].apply(lambda x: (set(sq1) & set(x)) == set(sq1))
    return ds

dataset = flag_sq(dataset.copy(), "title")

conditions = [dataset['hold'] == True, dataset['sell'] == True, dataset['buy'] == True]
choices = ['hold', 'sell', 'buy']
dataset["signal"] = np.select(conditions, choices, default="None")
dataset["signal"] = np.where((dataset.hold & dataset.sell) |
                             (dataset.hold & dataset.buy) |
                             (dataset.sell & dataset.buy), 'multiple', dataset.signal)
dataset["signal"] = np.where((dataset.signal == "None") & (dataset.rocket), 'rocket', dataset.signal)
dataset["signal"] = np.where((dataset.signal == "None") & (dataset.diamond), 'diamond', dataset.signal)

dataset = dataset[["real_datetime", "contains_gme", "signal", "squeeze"]] 
dataset.to_csv("C:/Users/karkl/Desktop/Master Thesis ESCP/viz/indicators/bhs3.csv")

"""      
signal = dataset[dataset.hold | dataset.sell | dataset.buy]
signal = signal[signal.hold ^ signal.sell ^ signal.buy]
signal = signal[signal["contains_gme"]]

conditions = [signal['hold'] == True, signal['sell'] == True, signal['buy'] == True]
choices = ['hold', 'sell', 'buy']
signal["signal"] = np.select(conditions, choices)

signal = signal.drop(columns = ['hold', 'sell', 'buy'])

signal.to_csv("C:/Users/karkl/Desktop/Master Thesis ESCP/viz/indicators/bhs.csv")
"""
