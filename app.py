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
        data = request.form['input']
        print("MBTI Personality type prediction:", data)  # todo del testing purposes only
        pred = predict.to_mbti(data, api)
    else:
        pred = None
    return render_template('result.html', prediction=str(pred))


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    print('WSGI serving at http://127.0.0.1:5000/')
    http_server.serve_forever()
