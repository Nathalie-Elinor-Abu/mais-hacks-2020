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

y_test = encoder.fit_transform(test['type'].values)
y_test = to_categorical(y_test)


# get an idea of the distribution of the text values
from collections import Counter
ctr = Counter(train['type'].values)
print('Distribution of Classes:', ctr)

# get class weights for the training data, this will be used data
y_train_int = np.argmax(y_train,axis=1)
cws = class_weight.compute_class_weight('balanced', np.unique(y_train_int), y_train_int)
print(cws)
