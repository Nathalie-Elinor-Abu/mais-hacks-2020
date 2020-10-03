import concurrent.futures
import json
import sys
import toml
import time
import tweepy
import datetime
import sqlite3
import traceback
from tweepy import RateLimitError, TweepError
import tweepy
from tweepy import Cursor
import multiprocessing
import tweetAPIKEYS


# Auth the API keys for Tweepy.
def auth(consumer_key, consumer_secret, access_token, access_token_secret):
    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        return tweepy.API(auth)
    except:
        print("authentication failed")
        inputedval = input("would you like to terminate the entire program? (Y/N)")
        inputedval.lower()
        if (inputedval == "y") or (inputedval == "yes"):
            exit(1)
        return None

# fix this
auth_api = auth(tweetAPIKEYS.consumer_key, tweetAPIKEYS.consumer_secret, tweetAPIKEYS.access_token, tweetAPIKEYS.access_token_secret)

def get_tweets_array(target,num_tweets):
    array = []
    count = 0
    for status in Cursor(auth_api.user_timeline, id=target,tweet_mode="extended").items():
        if count == num_tweets:
            break
        print(status.full_text)
        array = array + [status.full_text]
        count   = count + 1
    return array


for x in get_tweets_array(50393960, 20):
    print(x)
