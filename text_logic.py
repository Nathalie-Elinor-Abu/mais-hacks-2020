import tweepy
import tweetAPIKEYS



def to_mtbi(string,api):
    if string == None:
        return

    screen_name = get_screenname(string)
    final_str = ""
    count = 0
    for status in tweepy.Cursor(api.user_timeline, screen_name=screen_name, tweet_mode="extended").items():
        count = count + 1
        if count == 50:
            break
        print(final_str)
        final_str = final_str + status.full_text

    return final_str




def get_screenname(string):
    if string == "":
        return ""
    array = string.split("/")
    screen_name = array[-1].split("?")[0]
    return screen_name
