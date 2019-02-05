#%%
import numpy as np
import pandas as pd
#import xgboost as xgb

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping

from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn import preprocessing

stop_words = stopwords.words('english')

#%% [markdown]
## Read data

#%%
df_train = pd.read_csv("./python/SpookyAuthorIdentification/input/train.csv")
df_test = pd.read_csv("./python/SpookyAuthorIdentification/input/test.csv")

#%%
#%%
## Word Vectorise
# download the GloVe vectors from http://nlp.stanford.edu/data/glove.840B.300d.zip
# , and unzip
#%%
embeddings_index = {}
f = open("./python/SpookyAuthorIdentification/model/glove.840B.300d.txt", encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#%%
# creat vectors for the whole sentences 
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())


#%%
X_train = df_train['text'].values
X_train_glove = [sent2vec(x) for x in X_train]
X_train_glove = np.array(X_train_glove)

#%%
X_test = df_test['text'].values
X_test_glove = [sent2vec(x) for x in X_test]
X_test_glove = np.array(X_test_glove)


#%%
# stadarise
scl = preprocessing.StandardScaler()
X_train_glove_scl = scl.fit_transform(X_train_glove)
X_test_glove_scl = scl.transform(X_test_glove)
#%%
lbl_enc = preprocessing.LabelEncoder()
y_train = lbl_enc.fit_transform(df_train['author'].values)
y_train_enc = to_categorical(y_train)

#%%
# modeling
model = Sequential()

model.add(Dense(300, input_dim=300, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(300, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

#%%
model.fit(X_train_glove_scl, y_train_enc, batch_size=64, epochs=5, verbose=2, 
          validation_split=0.1)

#%%
# prediction
y_pred = model.predict(X_test_glove_scl)
df_pred = pd.DataFrame({'id': df_test.loc[:, 'id'], 
                        'EAP': [x[0] for x in y_pred],
                        'HPL': [x[1] for x in y_pred],
                        'MWS': [x[2] for x in y_pred]},
                        columns=['id', 'EAP', 'HPL', 'MWS'])
#%%
df_pred.to_csv('./python/SpookyAuthorIdentification/output/submission.csv', index=False)

#%%
# LSTM model
# using keras tokenizer here
token = text.Tokenizer(num_words=None)
max_len = 70

token.fit_on_texts(list(X_train))
X_train_seq = token.texts_to_sequences(X_train)
X_test_seq = token.texts_to_sequences(X_test)

# zero pad the sequences
X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=max_len)

word_index = token.word_index

#%%
# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


#%%
# A simple LSTM with glove embeddings and two dense layers
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


#%%
model.fit(X_train_pad, y_train_enc, batch_size=512, epochs=100, verbose=2,
    validation_split=0.1)

#%%_prediction
y_pred = model.predict(X_test_pad)
df_pred = pd.DataFrame({'id': df_test.loc[:, 'id'], 
                        'EAP': [x[0] for x in y_pred],
                        'HPL': [x[1] for x in y_pred],
                        'MWS': [x[2] for x in y_pred]},
                        columns=['id', 'EAP', 'HPL', 'MWS'])

#%%
df_pred
#%%
df_pred.to_csv('./python/SpookyAuthorIdentification/output/submission.csv', index=False)


#%%
