import pickle
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

global vocab
clean_vocab = pd.read_csv("../data/clean_vocab.csv")
vocab = [str(s) for i, s in clean_vocab['0'].items()]
vocab = np.array(vocab)


model_FT = pickle.load(open('./model_FT.model', 'rb'))
model_IE = pickle.load(open('./model_IE.model', 'rb'))
model_JP = pickle.load(open('./model_JP.model', 'rb'))
model_NS = pickle.load(open('./model_NS.model', 'rb'))

# Cache the stop words for speed
stop_words = set(stopwords.words("english"))

# creates a matrix of token counts
token_counts = CountVectorizer(analyzer="word",
                               max_features=1500,
                               tokenizer=None,
                               preprocessor=None,
                               stop_words=stop_words,
                               max_df=0.7,
                               min_df=0.1)
X_count = token_counts.fit_transform(vocab)
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_count).toarray()



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


    tweets = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', tweets)

    tweets = re.sub("[^a-zA-Z]", " ", tweets)

    tweets = re.sub(' +', ' ', tweets)

    tweets = " ".join([lemmatiser.lemmatize(w.lower()) for w in tweets.split(' ') if w not in stop_words]).strip()

    for t in unique_type_list:
        tweets = tweets.replace(t, "")

    myXcnt = token_counts.transform(np.array(tweets.split()))
    X_tfidf = tfidf_transformer.transform(myXcnt).toarray()

    results = [model_FT.predict(X_tfidf)[0],
               model_IE.predict(X_tfidf)[0],
               model_JP.predict(X_tfidf)[0],
               model_NS.predict(X_tfidf)[0],
               ]
    b_Pers_list = [{0: 'I', 1: 'E'}, {0: 'N', 1: 'S'}, {0: 'F', 1: 'T'}, {0: 'J', 1: 'P'}]

    s = ""
    for i, l in enumerate(results):
        s += b_Pers_list[i][l]
    return s

s = '$hello i am elinor 234 2hi hi @you great! poof #wow https://you.com'
print(cleanit(s))