from transformers import DistilBertModel, DistilBertConfig
import tensorflow as tf
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



data = pd.read_csv("../data/clean_data.csv")
data = data.drop(columns='type')
train, test = train_test_split(data, random_state = 42, test_size=0.15)
train, val = train_test_split(train, random_state = 42, test_size=0.15)
print(test.shape,train.shape,val.shape)