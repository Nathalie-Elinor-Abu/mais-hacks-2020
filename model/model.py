from transformers import DistilBertModel, DistilBertConfig
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, classification_report, log_loss
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Flatten
from keras.layers import Dropout, Conv1D, GlobalMaxPool1D, GRU, GlobalAvgPool1D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


MAX_SEQ_LEN = 25
DEFAULT_BATCH_SIZE = 128


unprocessed_data = pd.read_csv("../data/expanded_data.csv")
data = pd.read_csv("../data/clean_data.csv")
# data, unprocessed_data = data.drop(columns='type'), unprocessed_data.drop(columns='type')
train, test = train_test_split(data, random_state = 42, test_size=0.15)
train, val = train_test_split(train, random_state = 42, test_size=0.15)
print(test.shape,train.shape,val.shape)
print('train:\n', type(train.cleaned.values), train.cleaned.values[0])

# tokenize the sentences
print('tokenizing')
tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(train.cleaned.values)
train_text_vec = tokenizer.texts_to_sequences(train.cleaned.values)
test_text_vec = tokenizer.texts_to_sequences(test.cleaned.values)
val_text_vec = tokenizer.texts_to_sequences(val.cleaned.values)

# pad the sequences
print('padding')
train_text_vec = pad_sequences(train_text_vec, maxlen=MAX_SEQ_LEN)
test_text_vec = pad_sequences(test_text_vec, maxlen=MAX_SEQ_LEN)
val_text_vec = pad_sequences(val_text_vec, maxlen=MAX_SEQ_LEN)

print('Number of Tokens:', len(tokenizer.word_index))
print("Max Token Index:", train_text_vec.max(), "\n")

print('Sample Tweet Before Processing:', train["cleaned"].values[0])
print('Sample Tweet After Processing:', tokenizer.sequences_to_texts([train_text_vec[0]]), '\n')

print('What the model will interpret:', train_text_vec[0].tolist())


# One Hot Encode Y values:
print('encoding y vals')
encoder = LabelEncoder()

y_train = encoder.fit_transform(train['type'].values)
y_train = to_categorical(y_train)
y_val = encoder.fit_transform(val['type'].values)
y_val = to_categorical(y_val)

y_test = encoder.fit_transform(test['type'].values)
y_test = to_categorical(y_test)


# get an idea of the distribution of the text values
from collections import Counter
ctr = Counter(train['type'].values)
print('Distribution of Classes:', ctr)

# get class weights for the training data, this will be used data
y_train_int = np.argmax(y_train,axis=1)
cws = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_int), y=y_train_int)
print(cws)
cws_dict = {i[0]:w for i,w in np.ndenumerate(cws)}
print("class weights dictionary:\n",cws_dict)



'''
# BASELINE STATS AND TESTS
print('Dominant Class: ', ctr.most_common(n = 1)[0][0])
print('Baseline Accuracy Dominant Class', (ctr.most_common(n = 1)[0][0] == test['type'].values).mean())

preds = np.zeros_like(y_test)
preds[:, 0] = 1
preds[0] = 1 #done to suppress warning from numpy for f1 score
print('F1 Score:', f1_score(y_test, preds, average='weighted'))

# Naive Bayse Baseline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
text_clf.fit(tokenizer.sequences_to_texts_generator(train_text_vec), y_train.argmax(axis=1))
predictions = text_clf.predict(tokenizer.sequences_to_texts_generator(test_text_vec))
print('Baseline Accuracy Using Naive Bayes: ', (predictions == y_test.argmax(axis = 1)).mean())
print('F1 Score:', f1_score(y_test.argmax(axis = 1), predictions, average='weighted'))'''


