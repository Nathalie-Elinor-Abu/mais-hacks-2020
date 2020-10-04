#general imports
import re
import numpy as np
import pandas as pd

# for cleaning data
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# for vectorizing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# for xgboost model
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance

# tune learning_rate
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# saving/loading model
import pickle

# plotting

#%matplotlib inline

# read data
data = pd.read_csv('../data/mbti.csv')
#print(data.head(10))

# split posts in each entry on the delimiter
posts = [post.split('|||') for post in data.head(2).posts.values]


def get_types(row):
    mbti_type = row['type']

    I, N, T, J = 0, 0, 0, 0

    if mbti_type[0] == 'I':
        I = 1
    elif mbti_type[0] == 'E':
        I = 0
    else:
        print('I-E incorrect')

    if mbti_type[1] == 'N':
        N = 1
    elif mbti_type[1] == 'S':
        N = 0
    else:
        print('N-S incorrect')

    if mbti_type[2] == 'T':
        T = 1
    elif mbti_type[2] == 'F':
        T = 0
    else:
        print('T-F incorrect')

    if mbti_type[3] == 'J':
        J = 1
    elif mbti_type[3] == 'P':
        J = 0
    else:
        print('J-P incorrect')
    return pd.Series({'IE': I, 'NS': N, 'TF': T, 'JP': J})


#add columns for type indicators because they are not evenly distributed
data = data.join(data.apply(lambda row: get_types(row), axis=1))
#print(data.head(5))

# assign binary values to the mbti type indicators
binary_types = {'I': 0, 'E': 1, 'N': 0, 'S': 1, 'F': 0, 'T': 1, 'J': 0, 'P': 1}
binary_types_list = [{0: 'I', 1: 'E'}, {0: 'N', 1: 'S'}, {0: 'F', 1: 'T'}, {0: 'J', 1: 'P'}]


# transform mbti to binary vector
def indicator_to_binary(personality):
    return [binary_types[type] for type in personality]


# transform binary vector to mbti personality
def binary_to_indicator(personality):
    s = ''
    for i, l in enumerate(personality):
        s += binary_types_list[i][l]
    return s


d = data.head(4)
list_personality_bin = np.array([indicator_to_binary(p) for p in d.type])
print(f'Binary MBTI list: {list_personality_bin}')

# preprocess

# remove mbti types from the posts
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                    'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

unique_type_list = [mbti.lower() for mbti in unique_type_list]

# lemmatize & stemmatize
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

# Cache the stop words for speed
stop_words = set(stopwords.words("english"))


def pre_process_data(data, remove_stop_words=True, remove_mbti_profiles=True):
    ls_mbti = []
    ls_posts = []
    wcl = len(data)
    i = 0

    for row in data.iterrows():
        i += 1
        if i % 500 == 0 or i == 1 or i == wcl:
            print(f'Processed {i} of {wcl} rows')

        # remove and clean comments
        entries = row[1].posts
        tmp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', entries)
        tmp = re.sub("[^a-zA-Z]", " ", tmp)
        tmp = re.sub(' +', ' ', tmp).lower()

        if remove_stop_words:
            tmp = " ".join([lemmatiser.lemmatize(w) for w in tmp.split(' ') if w not in stop_words])
        else:
            tmp = " ".join([lemmatiser.lemmatize(w) for w in tmp.split(' ')])

        if remove_mbti_profiles:
            for t in unique_type_list:
                tmp = tmp.replace(t, "")

        labeled_type = indicator_to_binary(row[1].type)
        ls_mbti.append(labeled_type)
        ls_posts.append(tmp)

    ls_posts = np.array(ls_posts)
    ls_mbti = np.array(ls_mbti)
    return ls_posts, ls_mbti


entry_list, mbti_list = pre_process_data(data, remove_stop_words=True)
print("entry_list aka clean vocabulary\n", type(entry_list), type(entry_list[0]), '\n\n',entry_list, '\n\n\n')
vocab = pd.DataFrame(entry_list).to_csv('../data/clean_vocab.csv', index=False)

print("Number of entries and MBTI types: ", entry_list.shape, mbti_list.shape)

# check
print(entry_list[0], mbti_list[0])

# creates a matrix of token counts
token_counts = CountVectorizer(analyzer="word",
                               max_features=1500,
                               tokenizer=None,
                               preprocessor=None,
                               stop_words=stop_words,
                               max_df=0.7,
                               min_df=0.1)

# learn the vocabulary dictionary and return term-document matrix
print("CountVectorizer...")
X_count = token_counts.fit_transform(entry_list)

# Transform the count matrix to a normalized tf or tf-idf representation
tfidf_transformer = TfidfTransformer()

print("Tf-idf...")
# Learn the idf vector (fit) and transform a count matrix to a tf-idf representation
X_tfidf = tfidf_transformer.fit_transform(X_count).toarray()



feature_names = list(enumerate(token_counts.get_feature_names()))
#print(feature_names, X_tfidf.shape)


type_indicators = ["IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) – Sensing (S)",
                   "FT: Feeling (F) - Thinking (T)", "JP: Judging (J) – Perceiving (P)"]


#print("MBTI 1st row: %s" % binary_to_indicator(mbti_list[0, :]))
#print("Y: Binarized MBTI 1st row: %s" % mbti_list[0, :])

# build xgboost model for mbti dataset

# posts in tf-idf representation
X = X_tfidf



# setup parameters for xgboost
param = {}

param['n_estimators'] = 200
param['max_depth'] = 2
param['nthread'] = 8
param['learning_rate'] = 0.1

# Let's train type indicator individually
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))
    filename = f'model_{type_indicators[l][:2]}.model'

    Y = mbti_list[:, l]

    # split data into train and test sets
    # seed = 7
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

    # fit model on training data
    model = XGBClassifier(**param).fit(X_train, y_train)
    pickle.dump(model, open(filename, 'wb'))
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("* %s Accuracy: %.2f%%" % (type_indicators[l], accuracy * 100.0))