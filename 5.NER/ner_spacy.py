# -*- coding: utf-8 -*-
"""
April 2021
Comment: This file creates a NER system using the Spacy package (same as in the example Jupyter Notebook)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import spacy
from collections import Counter
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

PATH = ""

dataset = pd.read_csv(PATH + "submissions.csv")

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


dataset = keep_ascii(dataset.copy(), "title") 
dataset = remove_trailwhitespace(dataset.copy(), "title") 

#Next line takes 1 hour+ to run
dataset['org'] = dataset["title"].apply(lambda x: [str(word).lower() for word in nlp(str(x)).ents if word.label_=="ORG"])


"""#########"""
"""WORDCLOUD"""
"""#########"""

###TO PRODUCT THE FIVE WORDCLOUDS, CHANGE THE DATES IN THE TWO FOLLOWING LINES OR COMMENT FOR WHOLE SUBREDDIT.
#Iterate over months
filtered = dataset[dataset["real_datetime"]  >= np.datetime64("2021-02-01")]
filtered = filtered[filtered["real_datetime"]  <= np.datetime64("2021-02-28")]

counter = Counter()
top = filtered['org'].apply(lambda x: counter.update(x))

top100 = dict(counter.most_common(100))
my_wordcloud = WordCloud(background_color="white", width=1600, height=800, font_path='fonts/seguiemj.ttf').generate_from_frequencies(top100)
plt.figure(figsize=(20,10))
plt.axis("off")
#plt.title("All submissions" , fontsize=60)
plt.title("Submissions on February 2021", fontsize=60)
plt.imshow(my_wordcloud)
