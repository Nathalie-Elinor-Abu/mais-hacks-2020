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
from sklearn.model_selection import train_test_split


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


# preprocessing and cleaning the tweets

def clean(tweet):
    text = p.tokenize(tweet.lower())

    # stemming and lemmatization tools
    lemmer = nltk.stem.WordNetLemmatizer()
    w_tokenizer = TweetTokenizer()

    def lem(text):
        return [(lemmer.lemmatize(w)) for w in w_tokenizer.tokenize((text))]

    def remove_punct(text):
        new_words = []
        for word in text:
            # replace punctuation
            new_word = re.sub(r'[^\w\s]', '', (word))
            # make sure empty words are skipped
            if (new_word != '') and (new_word != None):
                new_words.append(new_word)
        return new_words

    return remove_punct(lem(text))


data_separated['preprocessed']=data_separated['tweet'].apply(clean)

# get rid of stop words
stop_words = set(stopwords.words('english'))
data_separated['cleaned'] = data_separated['preprocessed'].apply(lambda x: [item for item in \
                                                                            x if item not in stop_words])

# drop unnecessary columns, keeping only type,type_index,cleaned
data_separated = data_separated.drop(columns=['posts', 'tweet', 'preprocessed'])
print(data_separated.head(10))

# save to csv for ease of use
data_separated.to_csv('../data/clean_data.csv', index=False)