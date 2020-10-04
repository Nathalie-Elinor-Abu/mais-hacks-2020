from flask import Flask, request, render_template
from gevent.pywsgi import WSGIServer
import tweepy
from deploy import twitter_keys, predict

# authing the tweetpy while the app starts

auth = tweepy.OAuthHandler(twitter_keys.consumer_key, twitter_keys.consumer_secret)
auth.set_access_token(twitter_keys.access_token, twitter_keys.access_token_secret)
api = tweepy.API(auth)

app = Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
<<<<<<< HEAD
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
=======
        data = request.form['input']
        print("MBTI Personality type prediction:", data)  # todo del testing purposes only
        pred = predict.to_mbti(data, api)
    else:
        pred = None
    return render_template('result.html', prediction=str(pred))
>>>>>>> 654971c53056f0f4ec00c912269f181b947c0bac


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    print('WSGI serving at http://127.0.0.1:5000/')
    http_server.serve_forever()
