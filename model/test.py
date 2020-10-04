import pickle
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

global vocab
vocab = pd.read_csv("../data/clean_vocab.csv").to_numpy()
model_FT = pickle.load(open('./model_FT.model', 'rb'))
model_IE = pickle.load(open('./model_IE.model', 'rb'))
model_JP = pickle.load(open('./model_JP.model', 'rb'))
model_NS = pickle.load(open('./model_NS.model', 'rb'))
global tfidf_transformer
tfidf_transformer = pickle.load(open('tfidf_transformer.pkl', 'rb'))
global token_counts
token_counts = pickle.load(open('token_counts.pkl', 'rb'))

def cleanit(tweets):
    '''
    takes as input a string with all the tweets delimiter is whitespace
    returns array of strings
    '''
    unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                        'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

    unique_type_list = [mbti.lower() for mbti in unique_type_list]

    # lemmatize & stemmatize
    stemmer = PorterStemmer()
    lemmatiser = WordNetLemmatizer()

    # Cache the stop words for speed
    stop_words = set(stopwords.words("english"))

    word = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', tweets)
    word = re.sub("[^a-zA-Z]", " ", word)
    word = re.sub(' +', ' ', word).lower()
    word = " ".join([lemmatiser.lemmatize(w) for w in word.split(' ') if w not in stop_words]).strip()
    for t in unique_type_list:
        word = word.replace(t, "")

    '''# creates a matrix of token counts
    token_counts = CountVectorizer(analyzer="word",
                                   max_features=1500,
                                   tokenizer=None,
                                   preprocessor=None,
                                   stop_words=stop_words,
                                   max_df=0.7,
                                   min_df=0.1)
    X_count = token_counts.fit_transform(vocab)
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_count).toarray()'''

    return tfidf_transformer.transform(token_counts.transform(np.array(word))).toarray()

s = '$hello i am elinor 234 2hi hi @you great! poof #wow https://you.com'

result = model_FT.predict(cleanit(s))
print(result)