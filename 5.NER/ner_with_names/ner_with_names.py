# -*- coding: utf-8 -*-
"""
April 2021
Comment: This file creates a NER system using the list of stock names taken and calculates a table with the most popular occurences.
"""

import pandas as pd
import twokenize
import nltk
import spacy
from collections import Counter
nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')


SUBMISSION_PATH = ""
STOCK_CRYPTO_PATH = ""

dataset = pd.read_csv(SUBMISSION_PATH + "submissions.csv")

"""#############"""
"""PREPROCESSING"""
"""#############"""

def keep_ascii(ds, col):
    from string import ascii_letters
    allowed = set(ascii_letters + ' ')
    
    ds[col] = ds[col].apply(lambda x: ''.join(l for l in str(x) if l in allowed))
    return ds

def remove_trailwhitespace(ds, col):
    ds[col] = ds[col].apply(lambda x: ' '.join([token for token in x.split()]))
    return ds

def lower_case(ds, col):
    ds[col] = ds[col].apply(lambda x: str(x).lower())
    return ds

dataset = lower_case(dataset.copy(), "title") #short
dataset = keep_ascii(dataset.copy(), "title") 
dataset = remove_trailwhitespace(dataset.copy(), "title") 

def tokenize_without_stopwords(ds, col):
    stop_words_nltk = set(nltk.corpus.stopwords.words('english'))   
    func = (lambda x: [i for i in twokenize.tokenizeRawTweetText(x) if not i in stop_words_nltk])
    ds[col] = ds[col].apply(func)
    return ds

dataset = tokenize_without_stopwords(dataset.copy(), "title")


"""########################"""
"""Extracting stocks' names"""
"""########################"""
stocks = pd.read_csv(STOCK_CRYPTO_PATH + "stock_names.csv")
stocks = list(stocks["ticker"].to_numpy().flatten())

cryptos = pd.read_csv(STOCK_CRYPTO_PATH + "crypto_names.csv")[:100] #we only take the 100 most capitalized coins in this list to avoid very rare occurences.
cryptos = list(cryptos["ticker"].to_numpy().flatten())

ner_list = pd.DataFrame(stocks + cryptos, columns=["word"])
ner_list = lower_case(ner_list .copy(), "word") 
ner_list = remove_punctuation(ner_list .copy(), "word")["word"].to_list()


"""########"""
"""ANALYSIS"""
"""########"""
dataset['tickers'] = dataset["title"].apply(lambda x: [word for word in x if word in ner_list]) #quite long to run

#all dataset does not work well. We have a lot of words, for example the crypto MOON.
#However MOON means "to the moon" on the forum, not the crypto.
counter = Counter()
top = dataset['tickers'].apply(lambda x: counter.update(x))
top100 = dict(counter.most_common(1000))
