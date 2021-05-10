# -*- coding: utf-8 -*-
"""
UNUSED IN THE FINAL VERSION.
April 2021
Comment: This program implements pre-processing on the comment dataset and was useful to test the different preprocessing functions.
However functions are not optimal, some useless (replaced by Python native objects for example). 
I also did not implement a local path system and is not very commented. Use this file only if you want to see first analysis conducted.
"""

'''
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("C:/Users/karkl/Desktop/Master Thesis ESCP/dataset_1/filtered_bydate.csv")
dataset['title'] = dataset["permalink"].apply(lambda x: x[34:-9].replace("_", " "))
dataset['texte'] = dataset["body"] + " " + dataset["title"]


#judgment, #game stop is ignored
test = dataset[dataset["texte"].str.contains("gme|gamestop", case=False, na=False)]

date_count_full = (pd.to_datetime(dataset['created_utc'], unit='s').dt.floor('d').value_counts().rename_axis('date').reset_index(name='count'))
date_count_full = date_count_full.sort_values("date")
#plt.plot(date_count_full["date"], date_count_full["count"])

#Export proportion gme / gamestop vs gme price

date_count_gme = (pd.to_datetime(test['created_utc'], unit='s').dt.floor('d').value_counts().rename_axis('date').reset_index(name='count'))

date_count_full = (pd.to_datetime(dataset['created_utc'], unit='s').dt.floor('d').value_counts().rename_axis('date').reset_index(name='count'))

date_comp = date_count_full.merge(date_count_gme, left_on = "date", right_on = "date")
date_comp["proportion"] = date_comp["count_y"] / date_comp["count_x"]
date_comp = date_comp.sort_values("date")
plt.plot(date_comp["date"], date_comp["proportion"])

gme = pd.read_csv("C:/Users/karkl/Desktop/Master Thesis ESCP/gme_stock.csv")

gme = gme[["timestamp", "adjusted_close"]]
gme["timestamp"] = pd.to_datetime(gme["timestamp"])

date_comp = date_comp.merge(gme, left_on = "date", right_on = "timestamp")

date_comp.to_csv("C:/Users/karkl/Desktop/Master Thesis ESCP/viz/proportioncomment_vs_stock.csv")


#Create the popular word chart
def lower_case (ds, col):
    ds[col] = ds[col].apply(lambda x: x.lower())
    return ds

import re
def remove_digits(ds, col):
    ds[col] = ds[col].apply(lambda x: re.sub(r'\d+','', x))
    return ds


#removing punctuations
import string
def remove_punctuation(ds, col):
    ds[col] = ds[col].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    return ds

#import re
def remove_emojis(ds, col):
    #elist emojis
    emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)
    
    ds[col] = ds[col].apply(lambda x: emoji_pattern.sub(r'', x))
    return ds
    
def remove_trailwhitespace(ds, col):
    ds[col] = ds[col].apply(lambda x: ' '.join([token for token in x.split()]))
    return ds



def keep_ascii(ds, col):
    from string import ascii_letters
    allowed = set(ascii_letters + ' ')
    
    ds[col] = ds[col].apply(lambda x: ''.join(l for l in x if l in allowed))
    return ds

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
def tokenize_without_stopwords(ds, col):
    stop_words_nltk = set(stopwords.words('english'))   
    func = (lambda x: [i for i in word_tokenize(x) if not i in stop_words_nltk])
    ds[col] = ds[col].apply(func)
    return ds


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
def lemmatize(ds, col):
    lemmatizer = WordNetLemmatizer()
    ds[col] = ds[col].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    return ds
    
     

test = lower_case(test, "texte")
test = keep_ascii(test, "texte")
test = remove_trailwhitespace(test,"texte")
test = tokenize_without_stopwords(test,"texte")
test = lemmatize(test,"texte")
 
#test = remove_digits(test, "texte")
#test = remove_punctuation(test, "texte")
#test = remove_emojis(test, "texte")

dataset["texte"] = dataset["texte"].astype(str)

dataset = lower_case(dataset, "texte") #short
dataset = keep_ascii(dataset, "texte") #long
dataset = remove_trailwhitespace(dataset,"texte")
dataset = tokenize_without_stopwords(dataset,"texte") #very long
dataset = lemmatize(dataset,"texte")


dataset.to_csv("C:/Users/karkl/Desktop/Master Thesis ESCP/dataset_1/processed_text.csv")


dataset = pd.read_csv("C:/Users/karkl/Desktop/Master Thesis ESCP/dataset_1/processed_text.csv")
# when flooring, split at 00h00 while market only active
#test["created_utc"] = pd.to_datetime(test['created_utc'], unit='s').dt.floor('d')

dataset["created_utc"] = pd.to_datetime(dataset['created_utc'], unit='s').dt.floor('d')

import gc
def count_words_by_dayin_ctn(ds, col, ctn): #very long to run, the print gives an overview of runtime
    #select distinct days by timestamp
    output = pd.DataFrame()
    dates = pd.to_datetime(np.unique(ds[ctn], axis=0).tolist())
    for i in dates:
        print(i)
        temp = ds[ds[ctn] == i]
        temp = temp[col].apply(lambda x: pd.value_counts(x)).sum(axis = 0)
        temp = temp.sort_values(ascending=False)
        temp = pd.DataFrame(temp.head(1000), columns=[i])
        output = pd.concat([output, temp], axis=1)
        del temp
        gc.collect() #to avoid memory overflow
    return output
        
top1000_words = count_words_by_dayin_ctn(dataset, "texte", "created_utc")

dataset.to_csv("C:/Users/karkl/Desktop/Master Thesis ESCP/dataset_1/top1000_words.csv")
'''
