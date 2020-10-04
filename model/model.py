from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras import layers
from tensorflow import keras

# -------------------------------------------------------
from transformers import DistilBertModel, DistilBertConfig
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, classification_report, log_loss
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix



data = pd.read_csv("../data/clean_data.csv")
train, test = train_test_split(data, random_state = 42)
train, val = train_test_split(train, random_state = 42)

print(test.shape,train.shape,val.shape)
print('train:\n', type(train.tweet.values), train.tweet.values[0])

vocab_size = 10000
trunc_type = "post"
pad_type = "post"
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(data.tweet.values)
maxlen = 1500
train_sequences = tokenizer.texts_to_sequences(train.tweet.values)
train_padded = pad_sequences(train_sequences, maxlen = maxlen, truncating = trunc_type, padding = pad_type)

val_sequences = tokenizer.texts_to_sequences(val.tweet.values)
val_padded = pad_sequences(val_sequences, maxlen = maxlen, truncating = trunc_type, padding = pad_type)
print(train_padded)

# onehot encoding of labels
one_hot_labels = tf.keras.utils.to_categorical(train.type_index.values, num_classes=16)
val_labels= tf.keras.utils.to_categorical(val.type_index.values, num_classes=16)

#creating the Model!

def create_model():
    op = tf.keras.optimizers.Adam(learning_rate=0.00001)

    model = Sequential()
    model.add(Embedding(vocab_size, 256, input_length=maxlen-1))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(20)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])
    return model

# using TPU can reduce time spent training
use_tpu = False
if use_tpu:
    # Create distribution strategy
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # Create model
    with strategy.scope():
        model = create_model()
else:
    model = create_model()

print(model.summary())

# Running/fitting the model
model.fit(train_padded, one_hot_labels, epochs =20, verbose = 1,
          validation_data = (val_padded, val_labels),  callbacks = [tf.keras.callbacks.EarlyStopping(patience = 3)])
