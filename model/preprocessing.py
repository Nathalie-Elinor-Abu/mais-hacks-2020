import numpy as np
import pandas as pd
import re
import string
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


# cleaning the tweets
def clean_text(text):
    regex = re.compile('[%s]' % re.escape('|'))
    text = regex.sub(" ", text)
    words = str(text).split()
    words = [i.lower() + " " for i in words]
    words = [i for i in words if not "http" in i]
    words = " ".join(words)
    words = words.translate(words.maketrans('', '', string.punctuation))
    return words

data['tweet'] = data['posts'].apply(clean_text)
print("Sample row:\n", data.tweet.values[0])
print(data.head())

# drop unnecessary columns, keeping only type,type_index,cleaned
data = data.drop(columns=['posts'])
print(data)

# save to csv for ease of use
data.to_csv('../data/clean_data.csv', index=False)