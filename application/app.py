from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import pickle

# Use pickle to load in the pre-trained model.
#todo with open(f'model/mbti_model.pkl', 'rb') as f:
#    model = pickle.load(f)

app = Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET'])
def about():
    return render_template('index.html')


@app.route('/twitter-links')
def twitter():
    return render_template('links.html')


@app.route('/cp-text')
def text():
    return render_template('cp-text.html')


@app.route('/predict', methods=['POST'])
def output():
    return render_template('result.html')


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    print('WSGI Serving at http://127.0.0.1:5000/')
    http_server.serve_forever()

