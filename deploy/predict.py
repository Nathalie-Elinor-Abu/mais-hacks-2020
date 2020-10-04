import tweepy
from model import classification


def to_mbti(string, api):
    if string is None:
        return 'Error'
    elif 'http' in string:
        screen_name = get_screenname(string)
        final_str = ""
        count = 0
        for status in tweepy.Cursor(api.user_timeline, screen_name=screen_name, tweet_mode="extended").items():
            count += 1
            if count == 10:
                break
            print(final_str)
            final_str += status.full_text
    else:
        final_str = string

    mbti = classification.classify(final_str)

    return mbti


def get_screenname(string):
    if string == "":
        return ""
    array = string.split("/")
    screen_name = array[-1].split("?")[0]
    return screen_name
