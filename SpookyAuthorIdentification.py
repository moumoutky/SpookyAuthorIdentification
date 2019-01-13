#%%
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from ggplot import *
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import re

nltk.download('punkt')
nltk.download('stopwords')
porter = PorterStemmer()

#%% [markdown]
## Read data

#%%
df_train = pd.read_csv("./python/SpookyAuthorIdentification/input/train.csv")
df_test = pd.read_csv("./python/SpookyAuthorIdentification/input/test.csv")

#%% [markdown]
## EDA

#%%
df_train.head()

#%%
df_test.head()

#%%
len(df_train)

#%%
len(df_test)

#%%
df_train.isnull().any()

#%%
df_test.isnull().any()

#%% [markdown]
# -> no nan

#%%
ggplot(df_train, aes(x='author')) + \
geom_bar() + \
theme_bw()

#%% [markdown]
# -> It's not very inballance.

#%%
num_graph_tokens = 100 # the number of tokens in token count graph
list_author = list(df_train['author'].drop_duplicates())

#%%
for author in list_author:
    # tokenize
    text_series = df_train[df_train.author == author]['text']
    token_list = [nltk.word_tokenize(text) for text in text_series]

    # concat token list
    token_all = []
    for token in token_list:
        token_all = token_all + token 
    
    # count tokens
    num_token_all = Counter(token_all)
    num_token_all = sorted(num_token_all.items(), key=lambda x: x[1],  reverse=True)
    num_token_sentence = [len(token) for token in token_list]

    # plot frequency of tokens in texts by an author
    plt.figure(figsize=(9, 16))
    plt.title(author)
    plt.barh([num_token_all[num_graph_tokens-x][0] for x in range(num_graph_tokens)], 
             [num_token_all[num_graph_tokens-x][1] for x in range(num_graph_tokens)],)

    # plot histogram of the number of tokens in  a sentence
    plt.figure(figsize=(16, 9))
    plt.title(author)
    plt.hist(num_token_sentence)

#%% [markdown]
# frequency graphs -> removing stop words might work
# There are sentences including huge size of tokens in MWS.

#%% [markdown]
## Preprocessing for natural language
# define function
#%%
# cleansing (no use)
def cleansing_text(text):
    # remove punctuation
    dst = re.sub(r'[^\w\s]','',text) 
    dst = dst.lower() 
    return dst

# stop words removal (no use)
def remove_stopwords(text):
    stop = stopwords.words('english')
    return [word for word in text if word not in stop]

# tokenize
def tokenizer(text):
    return text.split()

# tokenize & steming by porter algorithm
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


#%% [markdown]
## Modeling

#%%
lr_tfidf = Pipeline([('vect',  TfidfVectorizer()),
                     ('clf', LogisticRegression(random_state=0))])

# grid search paramete
param_grid = [{ 'vect__stop_words': [stopwords.words('english'), None],
                'vect__tokenizer': [tokenizer, tokenizer_porter],
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [1, 10, 100]},
               {'vect__stop_words': [stopwords.words('english'), None],
                'vect__tokenizer': [tokenizer, tokenizer_porter],
                'vect__use_idf': [False],
                'vect__norm': [None],
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [1, 10, 100]}]

gs_lr_tfidf = GridSearchCV(lr_tfidf,
                            param_grid,
                            scoring='accuracy',
                            cv=4,
                            verbose=1,
                            n_jobs=-1)

X_train = df_train.loc[:, 'text']
y_train = df_train.loc[:, 'author']
gs_lr_tfidf.fit(X_train, y_train)

#%% [markdown]
## Evaluate model & prediction

#%%
print(gs_lr_tfidf.best_score_)
print(gs_lr_tfidf.best_params_)

#%%
clf = gs_lr_tfidf.best_estimator_
X_test = df_test.loc[:, 'text']
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)

df_pred = pd.DataFrame({'id': df_test.loc[:, 'id'], 
                        'EAP': [x[0] for x in y_pred_prob],
                        'HPL': [x[1] for x in y_pred_prob],
                        'MWS': [x[2] for x in y_pred_prob]},
                        columns=['id', 'EAP', 'HPL', 'MWS'])

#%%
# output
df_pred.to_csv('./python/SpookyAuthorIdentification/output/submission.csv')


#%%
