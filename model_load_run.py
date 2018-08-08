# general dependencies
import pickle
import string
import re
import numpy as np
import pandas as pd

# NLTK dependencies
from nltk.corpus import stopwords
from nltk import PorterStemmer

# sklearn dependencies
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

# defining the stopwords
stopword = stopwords.words('english')

#instantiating stemming object
ps = PorterStemmer()

#load the pickled countvectorizer
ngram_vect = joblib.load('ngram_vect.pkl')

#load the pickled gradientboost model
gb = joblib.load('gb.pkl')

#creating a clean text function
def cleanse(text):
    '''
    Function accepts a text input and does three things:
    1. Removes punctuation
    2. Splits into tokens
    3. Removes tokens that are stopwords, conducts stemming, and joins together into a single string
    '''
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = " ".join([ps.stem(word) for word in tokens if word not in stopword])
    return text

#delivering a prediction
def create_prediction(lyrics):

    # run cleanse function to remove punct, split into tokens, remove stopords, and perform stemming
    cleansed = cleanse(lyrics)

    #transform cleansed data into a 2d array
    arr = pd.Series(cleansed)
    
    #transform using the ngram vectorizer
    transformed_lyrics = ngram_vect.transform(arr)

    #generate a prediction
    prediction = gb.predict(transformed_lyrics)

    #return the prediction
    return prediction



    
    





