from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import numpy as np

RANDOM_SEED = 666

# fetch 20 newsgroups train dataset
train_data= fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes')) 

# fetch 20 newsgroups test dataset
test_data= fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes')) 


vectorizer = TfidfVectorizer(stop_words='english', min_df=0.001, max_df=0.20)
vectors_train = vectorizer.fit_transform(train_data.data)
vectors_test = vectorizer.transform(test_data.data)

X_train = vectors_train
y_train = train_data.target
X_test = vectors_test
y_test = test_data.target

model = LogisticRegression(class_weight="balanced",random_state=RANDOM_SEED)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

print('We get best precision for category 11 which is sci.crypt. This may be because it is a topic whose news data will have words which are not commonly used and are specific to cryptography. Thus it is easier for our model to differentiate crypt newsgroup vs other.')

print('We get lowest precision for category 19 which is talk.religion.misc. This may be because the misc news data will have words which are common to all the other news groups and there is no real differentiator for our model to learn to categorize this group.')
