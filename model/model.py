import numpy as np
import pandas as pd
import os

data = pd.read_csv("../data/mbti.csv")

# printing the first 5 lines
print(data.head(5))
print("Shape of the data", data.shape) # 8675 rows, 2 columns

# finding all the personality types
types = list(np.unique(data.type.values))
types_dict = {key: i for i, key in enumerate(types)}
print("There are", len(types), "personality types")
print(*types, sep=', ')

# assigning numbers to each type and putting it in the type_index column
def get_type_index(typ):
    return types_dict[typ]
data['type_index']=data['type'].apply(get_type_index)

# cleaning up and separating the tweets
def sep_tweets(tweets):
    return tweets.split('|||')
data['tweets_list']=data['posts'].apply(sep_tweets)
print(data.head())
data_separated = data.explode('tweets_list', ignore_index=True)
print(data_separated.shape)
