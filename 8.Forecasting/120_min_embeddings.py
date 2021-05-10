# -*- coding: utf-8 -*-
"""
May 2021
Comment: This file is used to create models at t+120 min with the embeddings. You have to manually change the variables tested in the code for the different models.
Also, I have included other metrics than AUC not presented in the thesis.
"""

import pandas as pd
import numpy as np
import pytz
import gensim
import nltk
import twokenize

PATH = ""

submissions = pd.read_csv(PATH + "submissions.csv")
gme_price = pd.read_csv(PATH + "stock_bymin.csv")
gme_price.drop(gme_price.tail(1).index,inplace=True) # drop last row
gme_price["time2"] = pd.to_datetime(gme_price["time"], format="%d/%m/%Y %H:%M")


eastern = pytz.timezone('US/Eastern')
gme_price.time2 = gme_price.time2.dt.tz_localize(eastern).dt.tz_convert(pytz.utc)
gme_price.time2 = gme_price.time2.dt.tz_localize(None)


gme_price["time"] = gme_price["time2"].apply(lambda x: x.timestamp())
gme_price = gme_price.sort_values(by=['time'], ascending=True)
#that's the last !!!! always

after = np.datetime64('2020-06-01 15:16:00')
after = (after - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's') 

gme_price[gme_price["time"]>=after]["time2"]

gme_price[gme_price["time"].gt(after)].iloc[0]["time2"]

#closer stockprice is the closest stock price recorded after the reddit submission. 
#If a message is posted at 15:16:34, it will return the stock price at 15:17:000
#If a message is posted at 23:45:00, it will return the first stock price recorded on the next day.
#extract["closer_stockprice"] = extract["created_utc"].apply(lambda x: gme_price[gme_price["time"].gt(x)].iloc[0]["time2"])
#1min:15



submissions["t0"] = pd.to_datetime(submissions["created_utc"], unit='s').dt.floor('min')
submissions = submissions.merge(gme_price[["time2", "close"]], how="left", left_on="t0", right_on="time2")
submissions["t120"] = submissions["t0"] + pd.DateOffset(hours=2)
submissions = submissions.merge(gme_price[["time2", "close"]], how="left", left_on="t120", right_on="time2")


submissions = submissions.drop(columns=["time2_x", "time2_y"])
submissions = submissions.dropna()
submissions["delta"] = (submissions["close_y"] / submissions["close_x"]) - 1
submissions["sign"] = np.where(submissions.delta >=0, 1, 0)
submissions["extreme"] = np.where(abs(submissions.delta) >= 0.1, 1, 0)



"""PREPROCESS"""
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

submissions = remove_apos_s(submissions.copy(), "title") 
submissions = keep_ascii(submissions.copy(), "title") 
submissions = remove_trailwhitespace(submissions.copy(), "title") 

submissions = lower_case(submissions.copy(), "title")

def tokenize_without_stopwords(ds, col):
    stop_words_nltk = set(nltk.corpus.stopwords.words('english'))   
    func = (lambda x: [i for i in twokenize.tokenizeRawTweetText(x) if not i in stop_words_nltk])
    ds[col] = ds[col].apply(func)
    return ds

submissions = tokenize_without_stopwords(submissions.copy(), "title")

"""SELECTION OF RELEVANT POSTS"""
submissions = submissions[submissions["title"].str.len()>0]
submissions = submissions[submissions["contains_gme"]]

"""MODELLING with the doc2vec"""

model = gensim.models.doc2vec.Doc2Vec.load(PATH + "doc2vec2.model") #change manually for the model to load

submissions["embeddings"] = submissions["title"].apply(lambda x: model.infer_vector(x))


#submissions_train = submissions[pd.to_datetime(submissions["real_datetime"]) < np.datetime64("2021-02-15")]
X = submissions.embeddings.apply(pd.Series)
Y = submissions[["delta", "sign", "extreme"]]

X_train = X[:69657]
Y_train = Y[:69657]

X_test = X[69657:]
Y_test = Y[69657:]

#regression bundle
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, Y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
r2_score(Y_train, clf.predict(X_train)) 

rms = sqrt(mean_squared_error(clf.predict(X_train), Y_train))
rms = sqrt(mean_squared_error(clf.predict(X_test), Y_test))

#classification bundle
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, Y_train["sign"])

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth = 5, verbose=1)
clf.fit(X_train, Y_train["sign"])

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=25, n_jobs=-1)
clf.fit(X_train, Y_train["sign"])

from sklearn.metrics import roc_auc_score
roc_auc_score(Y_train["sign"], clf.predict_proba(X_train)[:,1])
roc_auc_score(Y_test ["sign"], clf.predict_proba(X_test)[:,1])
               
from sklearn.metrics import accuracy_score
accuracy_score(Y_train["sign"], clf.predict(X_train))
accuracy_score(Y_test["sign"], clf.predict(X_test))



import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

model = keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
                   dtype=tf.string, input_shape=[], output_shape=[50]),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

history = model.fit(submissions["title"], submissions["extreme"], epochs=5)
