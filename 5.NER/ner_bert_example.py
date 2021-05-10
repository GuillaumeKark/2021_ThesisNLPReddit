# -*- coding: utf-8 -*-
"""
UNUSED IN THE FINAL VERSION BECAUSE OF PERFORMANCE.
April 2021
Comment: This file creates a NER system using the BERT Model with grouped entitites.
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
txt = "Saving every dime for blastoff BB to Mars!!! Start training for your career at the new AMC on The MOON!!"

ner_results = nlp(txt)
print(ner_results)  
