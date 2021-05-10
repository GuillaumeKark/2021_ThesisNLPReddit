# -*- coding: utf-8 -*-
"""
May 2021
Comment: This file is used to create embeddings models with Word2Vec. The notebook is configured for embeddings with submissions.
You can configure it for comments changing variable names and output name.
"""

import pandas as pd
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

dataset = pd.read_csv(PATH + "submissions.csv")

"""##############"""
"""PRE-PROCESSING"""
"""##############"""

def remove_apos_s(ds, col):
    ds[col] = ds[col].apply(lambda x: str(x).replace("'s", ""))
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

dataset = remove_apos_s(dataset.copy(), "title") 
dataset = keep_ascii(dataset.copy(), "title") 
dataset = remove_trailwhitespace(dataset.copy(), "title") 
dataset = lower_case(dataset.copy(), "title")

def tokenize_without_stopwords(ds, col):
    stop_words_nltk = set(nltk.corpus.stopwords.words('english'))   
    func = (lambda x: [i for i in twokenize.tokenizeRawTweetText(x) if not i in stop_words_nltk])
    ds[col] = ds[col].apply(func)
    return ds

dataset = tokenize_without_stopwords(dataset.copy(), "title")



dataset = dataset[["title", "created_utc", "contains_gme"]]

from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.1, random_state=42, shuffle=True)




train = list(train["title"])

for i in range(0,len(train)):
    train[i] =  gensim.models.doc2vec.TaggedDocument(train[i], [i])

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=20)

# Build a vocabulary
model.build_vocab(train)

print(f"Word 'rocket' appeared {model.wv.get_vecattr('rocket', 'count')} times in the training corpus.")

model.train(train, total_examples=model.corpus_count, epochs=model.epochs)


vector = model.infer_vector(["🚀"])
print(vector)

sims = model.dv.most_similar([vector], topn=len(model.dv))

for label, index in [('MOST', 0), ('SECOND', 1), ('THIRD', 2),('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train[sims[index][0]].words)))


model.save(PATH + "doc2vec.model")
