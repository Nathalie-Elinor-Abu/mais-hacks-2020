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

print("Number of entires and MBTI types: ", entry_list.shape, mbti_list.shape)

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

'''for mbti in range(len(type_indicators)):
    print(type_indicators[mbti])'''

#print("MBTI 1st row: %s" % binary_to_indicator(mbti_list[0, :]))
#print("Y: Binarized MBTI 1st row: %s" % mbti_list[0, :])

# build xgboost model for mbti dataset

# posts in tf-idf representation
X = X_tfidf

'''
# train type indicator individually
for l in range(len(type_indicators)):
    #print("%s ..." % (type_indicators[l]))

    Y = mbti_list[:, l]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("* %s Accuracy: %.2f%%" % (type_indicators[l], accuracy * 100.0))

# train type indicator individually
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))

    Y = mbti_list[:, l]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
    model = XGBClassifier()
    eval_set = [(X_test, y_test)]
    model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("* %s Accuracy: %.2f%%" % (type_indicators[l], accuracy * 100.0))


# Only the 1st indicator
y = mbti_list[:, 0]
# fit model on training data
model = XGBClassifier()
model.fit(X, y)
# plot feature importance
ax = plot_importance(model, max_num_features=25)

fig = ax.figure
fig.set_size_inches(15, 20)

#plt.show()

features = sorted(list(enumerate(model.feature_importances_)), key=lambda x: x[1], reverse=True)
for f in features[0:25]:
    print("%d\t%f\t%s" % (f[0], f[1], token_counts.get_feature_names()[f[0]]))

# Save xgb_params for late discussuin
default_get_xgb_params = model.get_xgb_params()

# Save xgb_params for later discussuin
default_get_xgb_params = model.get_xgb_params()
print(default_get_xgb_params)

# setup parameters for xgboost
param = {}

param['n_estimators'] = 200
param['max_depth'] = 2
param['nthread'] = 8
param['learning_rate'] = 0.2

# Let's train type indicator individually
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))

    Y = mbti_list[:, l]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("* %s Accuracy: %.2f%%" % (type_indicators[l], accuracy * 100.0))


# Posts in tf-idf representation
X = X_tfidf

# setup parameters for xgboost
param = {'n_estimators': 200, 'max_depth': 2, 'nthread': 8, 'learning_rate': 0.2}

# train type indicator individually
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))

    Y = mbti_list[:, l]
    model = XGBClassifier(**param)
    # learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    # param_grid = dict(learning_rate=learning_rate)

    param_grid = {
        'n_estimators': [200, 300],
        'learning_rate': [0.2, 0.3]
        # 'learning_rate': [ 0.01, 0.1, 0.2, 0.3],
        # 'max_depth': [2,3,4],
    }

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X, Y)

    # summarize results
    print("* Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("* %f (%f) with: %r" % (mean, stdev, param))



# A few few tweets and blog post
my_posts = """Getting started with data science and applying machine learning has never been as simple as it is now. There are many free and paid online tutorials and courses out there to help you to get started. I’ve recently started to learn, play, and work on Data Science & Machine Learning on Kaggle.com. In this brief post, I’d like to share my experience with the Kaggle Python Docker image, which simplifies the Data Scientist’s life.
Awesome #AWS monitoring introduction.
HPE Software (now @MicroFocusSW) won the platinum reader's choice #ITAWARDS 2017 in the new category #CloudMonitoring
Certified as AWS Certified Solutions Architect 
Hi, please have a look at my Udacity interview about online learning and machine learning,
Very interesting to see the  lessons learnt during the HP Operations Orchestration to CloudSlang journey. http://bit.ly/1Xo41ci 
I came across a post on devopsdigest.com and need your input: “70% DevOps organizations Unhappy with DevOps Monitoring Tools”
In a similar investigation I found out that many DevOps organizations use several monitoring tools in parallel. Senu, Nagios, LogStach and SaaS offerings such as DataDog or SignalFX to name a few. However, one element is missing: Consolidation of alerts and status in a single pane of glass, which enables fast remediation of application and infrastructure uptime and performance issues.
Sure, there are commercial tools on the market for exactly this use case but these tools are not necessarily optimized for DevOps.
So, here my question to you: In your DevOps project, have you encountered that the lack of consolidation of alerts and status is a real issue? If yes, how did you approach the problem? Or is an ChatOps approach just right?
You will probably hear more and more about ChatOps - at conferences, DevOps meet-ups or simply from your co-worker at the coffee station. ChatOps is a term and concept coined by GitHub. It's about the conversation-driven development, automation, and operations.
Now the question is: why and how would I, as an ops-focused engineer, implement and use ChatOps in my organization? The next question then is: How to include my tools into the chat conversation?
Let’s begin by having a look at a use case. The Closed Looped Incidents Process (CLIP) can be rejuvenated with ChatOps. The work from the incident detection runs through monitoring until the resolution of issues in your application or infrastructure can be accelerated with improved, cross-team communication and collaboration.
In this blog post, I am going to describe and share my experience with deploying HP Operations Manager i 10.0 (OMi) on HP Helion Public Cloud. An Infrastructure as a Service platform such as HP Helion Public Cloud Compute is a great place to quickly spin-up a Linux server and install HP Operations Manager i for various use scenarios. An example of a good use case is monitoring workloads across public clouds such as AWS and Azure.
"""

# The type is just a dummy so that the data prep fucntion can be reused
mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})

my_posts, dummy = pre_process_data(mydata, remove_stop_words=True)

my_X_cnt = token_counts.transform(my_posts)
my_X_tfidf = tfidf_transformer.transform(my_X_cnt).toarray()

# setup parameters for xgboost
param = {'n_estimators': 200, 'max_depth': 2, 'nthread': 8, 'learning_rate': 0.2}

result = []
# Let's train type indicator individually
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))

    Y = mbti_list[:, l]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)

    # make predictions for my  data
    y_pred = model.predict(X)
    result.append(y_pred[0])
    # print("* %s prediction: %s" % (type_indicators[l], y_pred))

print("The result is: ", binary_to_indicator(result))
'''

# setup parameters for xgboost
param = {}

param['n_estimators'] = 200
param['max_depth'] = 2
param['nthread'] = 8
param['learning_rate'] = 0.2

# Let's train type indicator individually
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))
    filename = f'model_{type_indicators[l][:2]}.model'

    Y = mbti_list[:, l]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
    model = XGBClassifier(**param).fit(X_train, y_train)
    pickle.dump(model, open(filename, 'wb'))
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("* %s Accuracy: %.2f%%" % (type_indicators[l], accuracy * 100.0))

