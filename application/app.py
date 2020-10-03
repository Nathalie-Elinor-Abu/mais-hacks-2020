import flask
import pickle
# Use pickle to load in the pre-trained model.
#todo with open(f'model/mbti_model.pkl', 'rb') as f:
#    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')


@app.route('/')
def main():
    return flask.render_template('main_page.html')


@app.route('/out/')
def output():
    return flask.render_template('output_page.html')


@app.route('/about/')
def about():
    return flask.render_template('about_page.html')


@app.route('/mbti-text/')
def text():
    return flask.render_template('about_page.html')


if __name__ == '__main__':
    app.run(debug=True)
