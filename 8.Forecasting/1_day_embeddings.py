# -*- coding: utf-8 -*-
"""
May 2021
Comment: This file is used to create models at t+1 day with the embeddings. You have to manually change the variables tested in the code for the different models.
Also, I have included other metrics than AUC not presented in the thesis.
"""

import pandas as pd
import gensim
import nltk
import twokenize
import emoji 
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

PATH = ""

submissions = pd.read_csv(PATH + "submissions.csv")
gme_price = pd.read_csv(PATH + "gme_stock.csv")

gme_price["timestamp"] = pd.to_datetime(gme_price["timestamp"], format="%d/%m/%Y")


#submissions = submissions[["title", "real_datetime"]]
#very long to perform
#submissions = submissions.groupby(by="real_datetime", dropna=False).sum()


"""PREPROCESS"""
def remove_apos_s(ds, col):
    ds[col] = ds[col].apply(lambda x: str(x).replace("'s", ""))
    ds[col] = ds[col].apply(lambda x: str(x).replace("’s", ""))
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

#removing punctuations
import string
def remove_punctuation(ds, col):
    ds[col] = ds[col].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation + "“”")))
    return ds


def remove_trailwhitespace(ds, col):
    ds[col] = ds[col].apply(lambda x: ' '.join([token for token in x.split()]))
    return ds

def lower_case(ds, col):
    ds[col] = ds[col].apply(lambda x: str(x).lower())
    return ds


submissions = remove_apos_s(submissions.copy(), "title") 
submissions = emoji_spacer(submissions.copy(), "title")
submissions = remove_punctuation(submissions.copy(), "title")
submissions = remove_trailwhitespace(submissions.copy(), "title") 
submissions = lower_case(submissions.copy(), "title")



def tokenize_without_stopwords(ds, col):
    stop_words_nltk = set(nltk.corpus.stopwords.words('english'))   
    func = (lambda x: [i for i in twokenize.tokenizeRawTweetText(x) if not i in stop_words_nltk])
    ds[col] = ds[col].apply(func)
    return ds

submissions = tokenize_without_stopwords(submissions.copy(), "title")


"""MODELLING with the doc2vec"""
submissions = submissions[submissions["title"].str.len()>0]

submissions = submissions[submissions["contains_gme"]]

model = gensim.models.doc2vec.Doc2Vec.load(PATH +"doc2vec.model") #the embeddings model to load. Change manually.

submissions["title"] = submissions["title"].apply(lambda x: model.infer_vector(x))

X = submissions.title.apply(pd.Series)
Y = submissions["delta"]

X_train = X[:70000]
Y_train = Y[:70000]

X_test = X[70000:]
Y_test = Y[70000:]

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
clf.fit(X_train, Y_train)

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(random_state=42, n_estimators=25, max_depth = 10, verbose=1)
clf.fit(X_train, Y_train)

from sklearn.metrics import roc_auc_score
roc_auc_score(Y_train, clf.predict_proba(X_train)[:,1])
roc_auc_score(Y_test, clf.predict_proba(X_test)[:,1])
               
from sklearn.metrics import accuracy_score
accuracy_score(Y_train, clf.predict(X_train))
accuracy_score(Y_test, clf.predict(X_test))


""" Testing Tensorflow embeddings with simple neural network model"""

model = keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
                   dtype=tf.string, input_shape=[], output_shape=[50]),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

history = model.fit(submissions["title"], submissions["extreme"], epochs=5)
