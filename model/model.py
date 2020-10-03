import numpy as np
import pandas as pd
import re
import preprocessor as p
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download
# nltk.download('wordnet')
# nltk.download('stopwords')
from nltk.corpus import wordnet
from nltk.tokenize import TweetTokenizer

data = pd.read_csv("../data/mbti.csv")

# printing the first 5 lines
print(data.head(5))
print("Shape of the data", data.shape)  # 8675 rows, 2 columns

# finding all the personality types
types = list(np.unique(data.type.values))
types_dict = {key: i for i, key in enumerate(types)}  # dict mapping type to int index
print("There are", len(types), "personality types")
print(*types, sep=', ')


# assigning numbers to each type and putting it in the type_index column
def get_type_index(typ):
    return types_dict[typ]


data['type_index'] = data['type'].apply(get_type_index)


# separating the tweets
def sep_tweets(tweets):
    return tweets.split('|||')


data['tweet'] = data['posts'].apply(sep_tweets)
data_separated = data.explode('tweet', ignore_index=True)
print("New shape of the data", data_separated.shape)


# preprocessing the tweets
def clean(tweet):
    tweets = p.tokenize(tweet.lower()).split()
    new_words = []
    for word in tweets:
        # replace punctuation
        new_word = re.sub(r'[^\w\s]', '', (word))
        # make sure empty words are skipped
        if new_word != '':
            new_words.append(new_word)
    return new_words


data_separated['preprocessed'] = data_separated['tweet'].apply(clean)

# stemming and lemmatization
lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = TweetTokenizer()


def lemmatize_text(text):
    return [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize((text))]


print(data_separated.head(10))
