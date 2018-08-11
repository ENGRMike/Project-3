"""
Machine Learning application for final Data Analytics project
© 2018
created by:
    Anna Bower
    Michael Bruins
    Seth Drewry
    Bobby Jaikaran
    Sam Stone
"""

# Application initiation packages
from flask import Flask, render_template, request
import flask_appbuilder as fa
from model import cleanse, create_prediction, search_lyrics

app = Flask(__name__)
@app.route('/')
def main():
    return render_template('j2_query.html')
    
@app.route('/process', methods=['POST'])
def lyrics():  # Retrieve the HTTP POST request parameter value from 'request.form' dictionary
        # get(attr) returns None if attr is not present      # Validate and send response
    _lyrics = request.form.get('lyrics')
    _header = create_prediction(_lyrics)[3]
    _data = create_prediction(_lyrics)[2]
    if lyrics:
        return render_template('j2_response.html', header=_header, data=_data)
    else:
        return 'Please go back and enter your name...', 400

@app.route('/api', methods=['POST'])
def api():
    _username = request.form.get('username')
    if _username:
        return render_template('j2_response_api.html', username=_username)
    else:
        return 'Please go back and enter your name...', 400
