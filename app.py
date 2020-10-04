from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import pickle
import tweepy
import tweetAPIKEYS
import text_logic
# authing the tweetpy while the app starts

auth = tweepy.OAuthHandler(tweetAPIKEYS.consumer_key, tweetAPIKEYS.consumer_secret)
auth.set_access_token(tweetAPIKEYS.access_token, tweetAPIKEYS.access_token_secret)
api = tweepy.API(auth)






# Use pickle to load in the pre-trained model.
#todo with open(f'model/mbti_model.pkl', 'rb') as f:
#    model = pickle.load(f)

app = Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST', 'GET'])
def prediction():
    pred = None
    if request.method == 'POST':
        pred = request.form['input']
        if pred.contains('http'):
            pass
            # todo pass to tweet extractor THEN to preprocess -> model
            # try - except "uove exceeded the allotted number of tweets for this time period, try copy/pasting your data instead
        else:
            pass
            # todo pass straight to preprocess -> model
        print("MBTI Personality type prediction:", pred)
    return render_template('result.html', prediction=(text_logic.to_mtbi(pred,api)))


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    print('WSGI serving at http://127.0.0.1:5000/')
    http_server.serve_forever()
