import pickle
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

global vocab
clean_vocab = pd.read_csv("data/clean_vocab.csv")
vocab = [str(s) for i, s in clean_vocab['0'].items()]
vocab = np.array(vocab)

model_FT = pickle.load(open('model/model_FT.model', 'rb'))
model_IE = pickle.load(open('model/model_IE.model', 'rb'))
model_JP = pickle.load(open('model/model_JP.model', 'rb'))
model_NS = pickle.load(open('model/model_NS.model', 'rb'))

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


def classify(input):
    unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                        'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

    unique_type_list = [mbti.lower() for mbti in unique_type_list]

    # lemmatize & stemmatize
    stemmer = PorterStemmer()
    lemmatiser = WordNetLemmatizer()

    input = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', input)

    input = re.sub("[^a-zA-Z]", " ", input)

    input = re.sub(' +', ' ', input)

    input = " ".join([lemmatiser.lemmatize(w.lower()) for w in input.split(' ') if w not in stop_words]).strip()

    for t in unique_type_list:
        input = input.replace(t, "")

    myXcnt = token_counts.transform(np.array(input.split()))
    X_tfidf = tfidf_transformer.transform(myXcnt).toarray()

    results = [model_IE.predict(X_tfidf)[0],
               model_NS.predict(X_tfidf)[0],
               model_FT.predict(X_tfidf)[0],
               model_JP.predict(X_tfidf)[0],
               ]

    print(model_FT.predict(X_tfidf),
               model_IE.predict(X_tfidf),
               model_JP.predict(X_tfidf),
               model_NS.predict(X_tfidf))
    print(results)

    b_Pers_list = [{0: 'I', 1: 'E'}, {0: 'N', 1: 'S'}, {0: 'F', 1: 'T'}, {0: 'J', 1: 'P'}]

    s = ""
    for i, l in enumerate(results):
        s += b_Pers_list[i][l]
    return s

print(classify('ENTJ,Hello!  I am working on a presentation by type.  Part of each presentation is feedback from a range of people of the type being reviewed.  I would greatly appreciate it if you could take a few...|||Mmmm...hmmmm.  Well, irritants:    People who bitch about everything.  Suck it up folks, live and learn or get out of the way.  When I can hear someone chew thier food.  That I am never sure if...|||I like to read and generally have 4-5 books going at once.  Planning...I plan trips months in advance and like to hone my spreadsheeted plan for fun :)  I also like to hike and be outside doing...|||Um, Hilter for sure.  And this lady I work with that everyone hates, we just did a workshop and she came up as my type...nice.|||Sweating :frustrating:  It is 95 degrees (F) out and we have no a/c nor windows that open in our office.  My office is currently a blamy 89!  I swear if one more person wants to have a closed door...|||If I think about it too long, it can really freak me out.  But I tend to be more of the buck up, germs make you stronger school.  Cleanliness, I want it to sparkle, but I dont want to be the...|||Or have said to yourself, after passing with the highest grade in the class, I cant believe I just faked my way though another one!  I have never once studied for a test in my life and I have...|||Well, thank goodness its not just me! Did you notice, with the exception of DearSig that it was mostly the men that said they are good at it...so I wonder if they truly are better at navigation or...|||I wonder if its because everyone is constantly trying to relay directions to me based on landmarks and where people used to live...|||What difference I see is:  Alpha role in work = leading to solve the problem  Alpha role in social situations = leading to make everyone happy and having fun  ...possibly.  For me, at least...|||1. The speaking to a large amount of people (30 vs 300) is also very difficult for me.  2. I am super impatient and this isnt good for parenting...or anything else.   3. Weakness 2 leads me to...|||I try to be a more green person but tend to agree with a some of the folks on here that a lot of the hoopla is just hype (global warming, melting icebergs, etc).  You just cant believe anything...|||Do any other ENTJs have a hard time navigating without a map?  I can get there find if I have a map an know I have to be the navigator, but otherwise, I am terrible about finding my way around a...|||I have to confess that I really have a deep love of lists...check lists, to-do lists, shopping lists.  I feel like writing these things down frees up space in my mind...usually to think up another...|||When there is a crisis at hand or I have a very long list of things to do.  If it get very bored, I canT'))