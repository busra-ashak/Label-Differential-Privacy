#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings("ignore")  # "error", "ignore", "always", "default", "module" or "once"


# In[3]:


import numpy as np
import random
import copy
#from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score as acc

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, ReLU
from tensorflow.keras.models import Model


# In[4]:


filename = 'bodmas.npz'
data = np.load('./' + filename)
X = data['X']  # all the feature vectors
y = data['y']  # labels, 0 as benign, 1 as malicious

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), test_size = 0.3, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # Train-Test

# In[5]:


classifiers = {"SGD": SGDClassifier(),"MLP": MLPClassifier(random_state=1, max_iter=300), "XGB": XGBClassifier(), "LGBM": LGBMClassifier()} 

print('Starting training...')

for classifier_pair in classifiers.items():
    print("---------------------------")
    print(classifier_pair[0])
    
    classifier = classifier_pair[1]
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    #compute accuracy_score
    accuracy = acc(y_test, y_pred)
    print('accuracy', accuracy)
    
print("---------------------------")


# # AutoKeras Model

# In[ ]:


saved_model=keras.models.load_model('structured_data_classifier/best_model')
print(saved_model.summary())


# In[ ]:


def autoKerasModel(path_best_model):
    saved_model = keras.models.load_model(path_best_model)
    input_layer = Input(shape=(2381,))
    x = saved_model.layers[1](input_layer)
    x = saved_model.layers[2](x)
    x = Dense(units=32)(x)
    x = ReLU()(x)
    x = Dense(units=32)(x)
    x = ReLU()(x)
    x = Dense(units=1)(x)
    x = saved_model.layers[-1](x)
    new_model = Model(inputs=input_layer, outputs=x)
    return new_model


# In[ ]:


autokeras_model=autoKerasModel('structured_data_classifier/best_model')
autokeras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

autokeras_model.fit(X_train, y_train, epochs=15)
results = autokeras_model.evaluate(X_test, y_test)

print("test loss, test acc:", results)
print(autokeras_model.summary())


# # Ensemble Learning - Majority Voting

# In[ ]:


def majority_voting(y_preds, y_test):
    assert y_preds.shape[0] == len(y_test), "y_preds's length is: {} while y_test's length is: {}. They should be equal.".format(y_preds.shape[0],len(y_test))
    y_pred_vote = []
    for preds in y_preds:
        if sum(preds) >= 3:
            y_pred_vote.append(1)
        else:
            y_pred_vote.append(0)
    #compute accuracy_score
    accuracy = acc(y_test, y_pred_vote)
    return accuracy


# In[ ]:


y_preds = np.ndarray(shape=(5,len(y_test)))
i=0
for classifier_pair in classifiers.items():
    classifier = classifier_pair[1]
    y_preds[i] = classifier.predict(X_test)
    i += 1
y_preds[i] = np.transpose(autokeras_model.predict(X_test, verbose=0))
y_preds = np.transpose(y_preds)
print('accuracy', majority_voting(y_preds, y_test))
