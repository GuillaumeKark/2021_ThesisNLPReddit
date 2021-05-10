# -*- coding: utf-8 -*-
"""
May 2021
Comment: This file is used to create models at t+1 day with the aggregated measures. You have to manually change the variables tested in the code.
Also, I have included other metrics than AUC not presented in the thesis.
"""

import pandas as pd
import numpy as np
import nltk
import twokenize
import emoji 
import tensorflow
from tensorflow import keras

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


"""Building an indicator table for 1 day forecast"""

submissions = submissions[submissions["title"].str.len()>0]


gme_price = gme_price[["timestamp", "adjusted_close", "volume", "variation"]]
gme_price['previous_date'] = gme_price["timestamp"].shift(-1)
gme_price = gme_price[gme_price["previous_date"] >= np.datetime64("2020-01-01 00:00:00")]




hold_voc = ["hold", "holding", "held", "holds", "diamonds", "diamond", "hand", "hands", "still", '💎']
sell_voc = ["sell", "sold", "selling", "resell", "sale", "sells"]
buy_voc = ["buy", "purchase", "buying", "purchased", "purchasing", "bought", "purchases", "buys", "moon", "rocket", "mars", '🚀']

gme_price["nb_submissions"] = None
gme_price["%gme"] = None
gme_price["%hold"] = None
gme_price["%sell"] = None
gme_price["%buy"] = None
gme_price["%GMEhold"] = None
gme_price["%GMEsell"] = None
gme_price["%GMEbuy"] = None

def create_flag(ds, col, lis, flag_name):
    ds[flag_name] = ds[col].apply(lambda x: (set(lis) & set(x)) != set())
    return ds


for i in range(0, len(gme_price)):
    before = gme_price.iloc[i,0]
    after = gme_price.iloc[i,4]
    temp = submissions[submissions["created_utc"] >= after.timestamp()]
    temp = temp[temp["created_utc"] <= before.timestamp()]
    gme_price.iloc[i,5] = len(temp)
   
    try:
        gme_price.iloc[i,6] = temp.contains_gme.value_counts()[True] / gme_price.iloc[i,5]
    except:
        gme_price.iloc[i,6] = 0
   
    temp = create_flag(temp.copy(), "title", hold_voc, "hold")
    temp = create_flag(temp.copy(), "title", sell_voc, "sell")
    temp = create_flag(temp.copy(), "title", buy_voc, "buy")
        
    try:
        gme_price.iloc[i,7] = temp.hold.value_counts()[True] / gme_price.iloc[i,5]
    except:
        gme_price.iloc[i,7] = 0
        
    try:
        gme_price.iloc[i,8] = temp.sell.value_counts()[True] / gme_price.iloc[i,5]
    except:
        gme_price.iloc[i,8] = 0  
        
    try:
        gme_price.iloc[i,9] = temp.buy.value_counts()[True] / gme_price.iloc[i,5]
    except:
        gme_price.iloc[i,9] = 0  
        
    temp = temp[temp["contains_gme"]]
        
    try:
        gme_price.iloc[i,10] = temp.hold.value_counts()[True] / gme_price.iloc[i,5]
    except:
        gme_price.iloc[i,10] = 0
        
    try:
        gme_price.iloc[i,11] = temp.sell.value_counts()[True] / gme_price.iloc[i,5]
    except:
        gme_price.iloc[i,11] = 0  
        
    try:
        gme_price.iloc[i,12] = temp.buy.value_counts()[True] / gme_price.iloc[i,5]
    except:
        gme_price.iloc[i,12] = 0  
       
    print(i)
    
    
gme_price['today_volume'] = gme_price["volume"].shift(-1)
gme_price['today_close'] = gme_price["adjusted_close"].shift(-1)

#Making the columns very clear
gme_price['predicting_date'] = gme_price["timestamp"]
gme_price['delta'] = gme_price["variation"]
gme_price["sign"] = np.where(gme_price.delta >=0, 1, 0)
gme_price["extreme_delta"] = np.where(abs(gme_price.delta) >=0.1, 1, 0)

gme_price = gme_price.drop(columns=["timestamp", "volume", "variation"])

gme_price = gme_price[gme_price["nb_submissions"] != 0].reset_index(drop=True)

gme_price.drop(gme_price.tail(1).index,inplace=True)

gme_price.sort_values(by=['previous_date'], inplace=True)

gme_price = gme_price.reset_index(drop=True)

""" BUILDING MODELS """

train = gme_price[gme_price["previous_date"] < np.datetime64("2021-01-01")] #Date to change according to the period checked.
test = gme_price[gme_price["previous_date"] >= np.datetime64("2021-01-01")]

confusion1 = train.corr()
confusion2 = test.corr()

train.sign.value_counts()
test.sign.value_counts()
train.extreme_delta.value_counts()
test.extreme_delta.value_counts()

#train = train.drop(columns=["previous_date", "predicting_date"])
train["previous_date"] = train["previous_date"].apply(lambda x: x.timestamp())
train["predicting_date"] = train["predicting_date"].apply(lambda x: x.timestamp())


Y_train = train[["delta", "sign", "extreme_delta", "adjusted_close"]].reset_index(drop=True)
X_train = train.drop(columns=["delta", "sign", "extreme_delta", "adjusted_close"]).reset_index(drop=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns= X_train.columns)


"""TEST"""
test["previous_date"] = test["previous_date"].apply(lambda x: x.timestamp())
test["predicting_date"] = test["predicting_date"].apply(lambda x: x.timestamp())


Y_test = test[["delta", "sign", "extreme_delta",  "adjusted_close"]].reset_index(drop=True)
X_test = test.drop(columns=["delta", "sign", "extreme_delta",  "adjusted_close"]).reset_index(drop=True)

X_test = pd.DataFrame(scaler.transform(X_test), columns= X_train.columns)


#X_train = X_train.drop(columns=["predicting_date", "previous_date", "%hold", "%GMEsell", "%GMEbuy", "%GMEhold"])
#X_test = X_test.drop(columns=["predicting_date", "previous_date", "%hold", "%GMEsell", "%GMEbuy", "%GMEhold"])

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, Y_train["sign"])

from sklearn.ensemble import RandomForestClassifier
clf=  RandomForestClassifier(n_estimators=1000, max_depth=2, verbose=1, random_state=41)
clf.fit(X_train, Y_train["extreme_delta"])
clf.fit(X_test, Y_test["extreme_delta"])

from sklearn.ensemble import GradientBoostingClassifier
clf=  GradientBoostingClassifier(n_estimators=1000, max_depth=2, verbose=1, random_state=42)
clf.fit(X_train, Y_train["extreme_delta"])

from sklearn.metrics import roc_auc_score
roc_auc_score(Y_train["extreme_delta"], clf.predict_proba(X_train)[:,1])
roc_auc_score(Y_test["extreme_delta"], clf.predict_proba(X_test)[:,1])

weights = pd.DataFrame([X_test.columns, clf.coef_[0].tolist()]).transpose()


from sklearn.metrics import plot_roc_curve
plot_roc_curve(clf, X_test, Y_test["extreme_delta"])

from sklearn.metrics import plot_precision_recall_curve
plot_precision_recall_curve(clf, X_test, Y_test["sign"])
  

from sklearn.metrics import accuracy_score
accuracy_score(Y_train["sign"], clf.predict(X_train))
accuracy_score(Y_test["extreme_delta"], clf.predict_proba(X_test)[:,1]>= 0.20)

from sklearn.metrics import f1_score
f1_score(clf.predict_proba(X_test)[:,1]>= 0.2, Y_test["sign"])

from sklearn.metrics import confusion_matrix
confusion_matrix(clf.predict_proba(X_test)[:,1]>= 0.38, Y_test["extreme_delta"])

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
cross_val_score(clf, X_train, Y_train["extreme_delta"], cv=2, scoring="roc_auc")
cross_val_score(clf, X_test, Y_test["extreme_delta"], cv=2, scoring="roc_auc")

"""Interpretation"""
weights = pd.DataFrame([X_test.columns, clf.feature_importances_]).transpose()

coefs = pd.DataFrame([Y_test["extreme_delta"], clf.predict_proba(X_test)[:,1]]).transpose()

model = keras.models.Sequential([
    keras.layers.Dense(12, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

optimizer = keras.optimizers.SGD(lr=1e-2)
model.compile(loss="binary_crossentropy",
              optimizer=optimizer,
              metrics=["AUC"])

history = model.fit(X_train, Y_train["extreme_delta"], epochs=10,
                    validation_data=(X_test, Y_test["extreme_delta"]))

roc_auc_score(Y_train["extreme_delta"], model.predict(X_train))
roc_auc_score(Y_test["extreme_delta"], model.predict(X_test))


"""Regression model"""

#regression bundle
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, Y_train["adjusted_close"])

from sklearn.metrics import r2_score
r2_score(Y_train["adjusted_close"], clf.predict(X_train)) 


from sklearn.metrics import r2_score
r2_score(Y_test["adjusted_close"], clf.predict(X_test)) 


coefs = pd.DataFrame([Y_test["adjusted_close"], clf.predict(X_test)]).transpose()
