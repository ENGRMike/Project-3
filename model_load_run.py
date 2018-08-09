# general dependencies
import pickle
import string
import re
import numpy as np
import pandas as pd

# musixmatch dependencies
from musixmatch import Musixmatch
from config import api_key

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

#load the API key for musixmatch
musixmatch = Musixmatch(api_key)

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
    
    predict_probability = gb.predict_proba(transformed_lyrics)

    #return the prediction
    return prediction, predict_probability

#delivering a prediction
def search_lyrics(user_input_lyrics, artist_name):
    
    try:
        #calling API
        result = musixmatch.track_search(q_track = user_input_lyrics, q_artist = artist_name, page_size = 100, page = 1, s_track_rating = "desc")

        #getting track_id, artist name, and track_name for first track
        track_id = result['message']['body']['track_list'][0]['track']['track_id']
        artist_name = result['message']['body']['track_list'][0]['track']['artist_name']
        track_name = result['message']['body']['track_list'][0]['track']['track_name']

        #getting lyrics information
        lyrics = musixmatch.track_lyrics_get(track_id)
        lyrics_package = lyrics['message']['body']['lyrics']
        lyrics_body =  lyrics_package["lyrics_body"][:-72]
        explicit = lyrics_package['explicit']

        #concatenating the artist name, track name and lyrics body
        arr = artist_name + track_name + lyrics_body

        #run prediction algorithm
        lyric_prediction = create_prediction(arr)

        #determine the prediction and the probability
        if lyric_prediction[0][0] == 0:
            prediction = "Not Explicit"
            probability = lyric_prediction[1][0][0]
        else:
            prediction = "Explicit"
            probability = lyric_prediction[1][0][1]

        #determine the actual value
        if explicit == 0:
            actual = "Not Explicit"
        else:
            actual = "Explicit"

        #determine if our prediction was correct
        if prediction == actual:
            comparison = "Correct"
        else:
            comparison = "Incorrect"

        #print out details
        print(f"Song named {track_name} by {artist_name} found.")
        print(f"We predict that this song is {prediction} with a probability of {probability}")
        print(f"Musixmatch reports that this song is {actual}")
        print(f"Our prediction was {comparison}")
        print(f"Song lyrics are as follows:")
        print(lyrics_body)

        #return the lyric_prediction
        return lyric_prediction
    
    except:
        print("did not find lyrics")
    
    





