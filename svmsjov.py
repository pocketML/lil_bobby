import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import re

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

df = pd.read_excel('data/train.xls')

def clean_data(df, col='Reviews'):
    for i, row in df.iterrows():
        sent = row[col]
        # remove html tags
        sent = cleanhtml(sent)
        # lowercase
        sent = sent.lower()
        # remove emails and websites
        sent = ' '.join([i for i in sent.split() if '@' not in i or 'www.' in i or 'http:' in i])

        df.at[i, 'col'] = sent

clean_data(df)