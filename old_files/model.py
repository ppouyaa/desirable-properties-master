
# coding: utf-8

# #### Modules

# In[9]:


import numpy as np
import pandas as pd
import random
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
random.seed(1000)
np.random.seed(100)
from pathlib import Path


# ### Variable declaration

# In[15]:


dataset_file_name = "input.xlsx"
candid_indexes = []
fileName = os.path.abspath(dataset_file_name)


# ### Classifier

# In[16]:


def trainModel(train_file,candidate_indexes,size=0.25):
    data_input_interim = pd.read_excel(train_file)
    data_input = data_input_interim.drop(candidate_indexes,axis=0)
    Y = data_input["Outcome"]
    data_input.drop("Outcome",axis=1,inplace=True)
    feature_cols = data_input.columns
    X = data_input[feature_cols]
    X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=size, random_state=0)

    clf = RandomForestClassifier(max_depth = 20, 
                                 min_samples_split=2, 
                                 n_estimators = 100, 
                                 random_state = 1)
    clf = clf.fit(X_train, Y_train)
    return clf


# In[17]:


clf_model = trainModel(fileName,candid_indexes,size=0.25)


# In[18]:


def pred(test_df):
   res = clf_model.predict_proba(test_df)
   return res

