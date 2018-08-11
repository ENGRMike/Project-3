import pickle
import string
import re
import numpy as np
import pandas as pd
import nltk
from musixmatch import Musixmatch
from config import api_key
from nltk.corpus import stopwords
from nltk import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
directory = "modelData/"

try:
    stopword = nltk.corpus.stopwords.words("english")
except:
    nltk.download("stopwords")
    stopword = nltk.corpus.stopwords.words("english")

ps = PorterStemmer()
ngram_vect = joblib.load(directory+"ngram_vect.pkl")
gb = joblib.load(directory+"gb.pkl")
musixmatch = Musixmatch(api_key)


def cleanse(text):
    text = "".join(
        [word.lower() for word in text if word not
            in string.punctuation])
    tokens = re.split('\W+', text)
    text = " ".join([ps.stem(word) for word in tokens if word not in stopword])
    return text


def create_prediction(lyrics):
    pd.set_option("display.max_colwidth", -1)

    cleansed = cleanse(lyrics)
    arr = pd.Series(cleansed)
    transformed_lyrics = ngram_vect.transform(arr)
    prediction = gb.predict(transformed_lyrics)
    if prediction == 0:
        result_ = "Xplicitation Result: Non-Explict"
    else:
        result_ = "Xplicitation Result: Non-Explict"
    predict_probability = gb.predict_proba(transformed_lyrics)
    n_expl = "{:.5%} probability that entry is non-explicit".format(
        predict_probability[0][0])
    expl = "{:.5%} probability that entry is explicit".format(
        predict_probability[0][1])
    w_breaks = lyrics.replace("\n", "\\r")
    df = pd.DataFrame({"-": [lyrics, n_expl, expl]}).set_index("-")
    output = df.style.set_properties(**{
        "background-color": "white",
        "text-align": "left",
    })
    output = str(
        df.to_html(index_names= False,
            justify = "left",
            classes = "table table-striped"
            )).replace("dataframe ","").replace("\\n", "<br/>").replace("\n","<br/>")
    return prediction, predict_probability, output, result_
    

def search_lyrics(title, artist):
    try:
        result = musixmatch.track_search(
            q_track=title,
            q_artist=artist,
            page_size=100,
            page=1,
            s_track_rating="desc")
        search = result["message"]["body"]["track_list"][0]
        track_id = search["track"]["track_id"]
        artist_name = search["track"]["artist_name"]
        track_name = search["track"]["track_name"]
        lyrics = musixmatch.track_lyrics_get(track_id)
        lyrics_package = lyrics["message"]["body"]["lyrics"]
        lyrics_body = lyrics_package["lyrics_body"][:-72]
        explicit = lyrics_package["explicit"]
        arr = artist_name + track_name + lyrics_body
        lyric_prediction = create_prediction(arr)
        if lyric_prediction[0][0] == 0:
            prediction = "Not Explicit"
            probability = lyric_prediction[1][0][0]
        else:
            prediction = "Explicit"
            probability = lyric_prediction[1][0][1]
        if explicit == 0:
            actual = "Not Explicit"
        else:
            actual = "Explicit"

        if prediction == actual:
            comparison = "Correct"
        else:
            comparison = "Incorrect"

        header = "Song \"{0}\" by {1} found.".format(track_name, artist_name)
        row_b = "We predict that this song is {} with {:.5%} accuracy confidence.\n\n".format(prediction, probability)
        row_c = "Musixmatch reports that this song is {0}".format(actual)
        row_d = "Our prediction was {0}".format(comparison)
        row_e = ""
        row_f = "Song lyrics are as follows:"
        row_g = lyrics_body

        api_df = pd.DataFrame({"-": [row_b, row_c, row_d, row_e, row_f, row_g]}).set_index("-")
        api_output = str(api_df.to_html(index_names=False,
                           justify="left", 
                           classes="table table-striped")
               ).replace("dataframe ","").replace("\\n", "<br/>").replace("\n","<br/>")
        # return the lyric_prediction 
    except:
        api_output = """<p style="font-size:80px;">&#x274C;</p>"""
        header ="did not find lyrics"
    return api_output, header
