import tweepy
import tweetAPIKEYS


def to_mtbi(string,api):
    id = get_id(string)
    status = api.get_status(id, tweet_mode="extended")
    return status.full_text


def get_id(string):
    array = string.split("/")
    return array[-1]


