import flask
import pickle

# Use pickle to load in the pre-trained model.
#todo with open(f'model/mbti_model.pkl', 'rb') as f:
#    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')


@app.route('/')
def about():
    return flask.render_template('index.html')


@app.route('/twitter-links/')
def main():
    return flask.render_template('links.html')


@app.route('/cp-text/')
def text():
    return flask.render_template('cp-text.html')


@app.route('/result/')
def output():
    return flask.render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)
