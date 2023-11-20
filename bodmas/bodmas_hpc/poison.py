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

# # Label Flipping

# In[ ]:


def attack_label_flipping(X_train, X_test, y_train, y_test, per, classifier_name, classifier):
    flipped_data = random_label_flipping(y_train, per)
    accuracy = 0
    if classifier_name == "AutoKeras":
        classifier.fit(X_train, flipped_data, verbose=0, epochs=15)
        y_pred = np.transpose(classifier.predict(X_test, verbose=0))
        accuracy = classifier.evaluate(X_test, y_test, verbose=0)[1]
    else:
        classifier.fit(X_train, flipped_data)
        y_pred = classifier.predict(X_test)
        accuracy = acc(y_test, y_pred)
    return y_pred, accuracy


# In[ ]:


#Flipping Random
def random_label_flipping(y_train, per):
    flip_count = int(per*(len(y_train)))
    flipped_data = copy.deepcopy(y_train)
    indices = random.sample(range(len(flipped_data)), flip_count)
    for j in indices:
        flipped_data[j] = (flipped_data[j] + 1)%2
    return flipped_data

#Flipping Specific
def specific_label_flipping(y_train, per, target):
    flipped_data = copy.deepcopy(y_train)
    possible_indices = []
    for i in range(len(y_train)):
        if y_train[i] == target:
            possible_indices.append(i)
    flip_count = int(per*(len(possible_indices)))
    indices = random.sample(possible_indices, flip_count)
    for j in indices:
        flipped_data[j] = (flipped_data[j] + 1)%2
    return flipped_data


# In[ ]:


percentages = [0.01, 0.05, 0.1, 0.2]

print("---------------------------")
for per in percentages:    
    poisoned_accuracies = {"SGD": [], "MLP": [], "XGB": [], "LGBM": [], "AutoKeras": []}
    ensemble_accuracies = []
    print( per, "poisoning starting...")
    for i in range(5):
        print("Trial #{} is starting...".format(i+1))
        
        poisoned_classifiers = {"SGD": SGDClassifier(), "MLP": MLPClassifier(random_state=1, max_iter=300), "XGB": XGBClassifier(), "LGBM": LGBMClassifier()} 
        poisoned_autokeras_model=autoKerasModel('structured_data_classifier/best_model')
        poisoned_autokeras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        poisoned_classifiers["AutoKeras"] = poisoned_autokeras_model
        
        y_preds = np.ndarray(shape=(len(poisoned_classifiers),len(y_test)))
        j=0
        for classifier_pair in poisoned_classifiers.items():
            poisoned_y_pred, poisoned_accuracy = attack_label_flipping(X_train, X_test, y_train, y_test, per, classifier_pair[0], classifier_pair[1])
            y_preds[j] = poisoned_y_pred
            j+=1
            poisoned_accuracies[classifier_pair[0]].append(poisoned_accuracy)
        y_preds = np.transpose(y_preds)
        
        ensemble_accuracy = majority_voting(y_preds, y_test)
        ensemble_accuracies.append(ensemble_accuracy)
        
        print("Trial #{} is completed with accuracy: {}".format(i+1, ensemble_accuracy))
        print("---------------------------")

    print("Random Poisoning - {}".format(per))
    for classifier_pair in poisoned_accuracies.items():
        accuracies = classifier_pair[1]
        print("{}'s average accuracy: {}".format(classifier_pair[0], sum(accuracies)/len(accuracies)))
    print("Average ensemble accuracy:", sum(ensemble_accuracies)/len(ensemble_accuracies))

