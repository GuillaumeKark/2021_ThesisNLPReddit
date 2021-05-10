# -*- coding: utf-8 -*-
"""
May 2021
Comment: This file is used to convert submissions to embeddings, visualizing T-SNE and some variables of interest.
"""

import pandas as pd
import numpy as np
import twokenize
import nltk
import spacy
nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

import gensim
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

PATH = ""

def remove_apos_s(ds, col):
    ds[col] = ds[col].apply(lambda x: str(x).replace("'s", ""))
    ds[col] = ds[col].apply(lambda x: str(x).replace("’s", ""))
    return ds

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


def tokenize_without_stopwords(ds, col):
    stop_words_nltk = set(nltk.corpus.stopwords.words('english'))   
    func = (lambda x: [i for i in twokenize.tokenizeRawTweetText(x) if not i in stop_words_nltk])
    ds[col] = ds[col].apply(func)
    return ds

"""#####################"""
"""IMPORTING SUBMISSIONS"""
"""#####################"""
test = pd.read_csv(PATH + "submissions.csv")

test = test[["title", "created_utc", "contains_gme"]]
test["real_datetime"] = pd.to_datetime(test["created_utc"], unit='s').dt.floor('d')

test = remove_apos_s(test.copy(), "title") 
test = keep_ascii(test.copy(), "title") 
test = remove_trailwhitespace(test.copy(), "title") 
test = lower_case(test.copy(), "title")
test = tokenize_without_stopwords(test.copy(), "title")


"""################"""
"""IMPORTING LABELS"""
"""################"""

gme_price = pd.read_csv(PATH + "gme_stock.csv")

gme_price["timestamp"] = pd.to_datetime(gme_price["timestamp"], format="%d/%m/%Y")


gme_price = gme_price[["timestamp", "adjusted_close", "volume", "variation"]]
gme_price['previous_date'] = gme_price["timestamp"].shift(-1)
gme_price = gme_price[gme_price["previous_date"] >= np.datetime64("2020-01-01 00:00:00")]

gme_price['today_volume'] = gme_price["volume"].shift(-1)
gme_price['today_close'] = gme_price["adjusted_close"].shift(-1)

#Making the columns very clear
gme_price['predicting_date'] = gme_price["timestamp"]
gme_price['delta'] = gme_price["variation"]
gme_price["sign"] = np.where(gme_price.delta >=0, 1, 0)
gme_price["extreme_delta"] = np.where(abs(gme_price.delta) >=0.1, 1, 0)

gme_price = gme_price.drop(columns=["timestamp", "volume", "variation"])
gme_price.drop(gme_price.tail(1).index,inplace=True)

gme_price.sort_values(by=['previous_date'], inplace=True)

gme_price = gme_price.reset_index(drop=True)


"""############################"""
"""MAPPING ALL DATES TO A DELTA"""
"""############################"""

test = test[test["real_datetime"] >= np.datetime64("2020-02-01")]
test = test[test["real_datetime"] <= np.datetime64("2021-04-01")]
uniques = pd.DataFrame(test.real_datetime.unique(), columns=["comment_date"])
uniques["date_previous"] = None
uniques["date_forecast"] = None
uniques["sign_label"] = None
uniques["extreme_label"] = None

for i in range(0, len(uniques)):
    temp = gme_price[gme_price["previous_date"] <= uniques.iloc[i,0]]
    temp = temp[temp["predicting_date"] > uniques.iloc[i,0]]
    uniques.iloc[i,1] = temp["previous_date"]
    uniques.iloc[i,2] = temp["predicting_date"]
    uniques.iloc[i,3] = temp["sign"]
    uniques.iloc[i,4] = temp["extreme_delta"]
    
uniques["date_previous"] = pd.to_datetime(uniques["date_previous"])

test = test.merge(uniques, how="left", left_on="real_datetime", right_on="comment_date")
#test = test.dropna()



"""############"""
"""FUNKY LABELS"""
"""############"""

hold_voc = ["hold", "holding", "held", "holds"]
sell_voc = ["sell", "sold", "selling", "resell", "sale", "sells"]
buy_voc = ["buy", "purchase", "buying", "purchased", "purchasing", "bought", "purchases", "buys"]
amc_voc = ["amc"]
tsla_voc = ["tsla", "tesla"]
pltr_voc = ["palantir", "pltr"]
diams_voc =  ["diamond","diamonds", "hand", "hands"]
moon_voc = ["moon", "moons"]
funds_voc = ["hedge", "funds", "fund", "melvin"]


def create_flag(ds, col, lis, flag_name):
    ds[flag_name] = ds[col].apply(lambda x: (set(lis) & set(x)) != set())
    return ds

test = create_flag(test.copy(), "title", hold_voc, "hold")
test = create_flag(test.copy(), "title", sell_voc, "sell")
test = create_flag(test.copy(), "title", buy_voc, "buy")
test = create_flag(test.copy(), "title", amc_voc, "amc")
test = create_flag(test.copy(), "title", tsla_voc, "tsla")
test = create_flag(test.copy(), "title", pltr_voc, "pltr")
test = create_flag(test.copy(), "title", diams_voc, "diamond")
test = create_flag(test.copy(), "title", moon_voc, "moon")
test = create_flag(test.copy(), "title", funds_voc, "funds")



from sklearn.utils import shuffle
test = shuffle(test, random_state=42)

Y = test.drop(columns=["title"])
X = test[["title"]]


sampleX = X[:25000]
sampleY = Y[:25000]
model = gensim.models.doc2vec.Doc2Vec.load(PATH + "doc2vec2.model") #loading the associated embeddings model 1 or 2
sampleX["embeddings"] = sampleX["title"].apply(lambda x: model.infer_vector(x))
sampleX = sampleX.drop(columns=["title"])
sampleX = sampleX.embeddings.apply(pd.Series)

from sklearn.manifold import TSNE
sampleX = TSNE(n_components=2, n_jobs=-1, verbose=1).fit_transform(sampleX)

output = pd.concat([pd.DataFrame(sampleX), sampleY.reset_index()], axis = 1)
output.to_csv(PATH + "tsne_model2.csv")
