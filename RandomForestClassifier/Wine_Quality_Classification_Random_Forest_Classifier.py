#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, classification_report


# In[2]:


cwd = os.getcwd()


# In[3]:


ip_filepath = os.path.dirname(cwd)+'\\WineQT.csv'


# In[4]:


wq_df = pd.read_csv(ip_filepath)


# In[5]:


print(wq_df.head())


# In[6]:


wq = wq_df.copy()


# In[7]:


wq.drop('Id',axis=1, inplace=True)
print("Dropped the Id column as it is an index column, and the df has already possess an index column")


# In[8]:


def replace_spaces(df):
    return df.columns.str.replace(" ","_")
def title_name(df):
    return df.columns.str.title()


# In[9]:


wq.columns = replace_spaces(wq)
wq.columns = title_name(wq)


# In[10]:


print(wq.columns)


# In[11]:


target = wq.loc[:,'Quality']
classifiers = wq.loc[:, wq.columns != 'Quality']


# #### Split train and test

# In[12]:


f_train, f_test, t_train, t_test = train_test_split(classifiers, target, test_size = 0.2, random_state = 101)
print(f'Shape of the f_train: {f_train.shape}')
print(f'Shape of the f_test: {f_test.shape}')
print(f'Shape of the t_train: {t_train.shape}')
print(f'Shape of the t_test: {t_test.shape}')


# #### Model Bulding and training

# In[13]:


# Model Instantiation:
rfc = RandomForestClassifier(random_state = 101)

# Training the model:
rfc_mod = rfc.fit(f_train, t_train)

# Prediction
rfc_pred = rfc_mod.predict(f_test)


# In[14]:


print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, rfc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, rfc_pred)*100,2)))


# #### Hyperparameter Tuning

# In[15]:


rfc = RandomForestClassifier(random_state = 101)


# In[16]:


params = {
    'bootstrap': [True],
    'max_depth': [5, 8, 10, 15],
    'max_features': [2, 3, 4, 5, 6],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [10, 20, 30, 50]
}


# In[17]:


grid_search = GridSearchCV(estimator=rfc, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=2, scoring = "accuracy")


# In[18]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(f_train, t_train)')


# In[19]:


grid_search.best_estimator_


# In[20]:


rfc = RandomForestClassifier(max_depth=15, max_features=3, min_samples_leaf=3,
                       min_samples_split=8, n_estimators=30, random_state=101)

# Training the model:
rfc_mod = rfc.fit(f_train, t_train)

# Prediction
rfc_pred = rfc_mod.predict(f_test)


# In[21]:


print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, rfc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, rfc_pred)*100,2)))


# #### There is no much improvement in the accuracy after hyperparameter tuning, however it performs well when compared to decision tree classifier

# ### Binary Classification

# In[22]:


target_binary = target.apply(lambda x: 1 if x>= 6 else 0)


# In[23]:


f_train, f_test, t_train, t_test = train_test_split(classifiers, target_binary, test_size = 0.2, random_state = 101)


# In[24]:


# Model Instantiation:
rfc = RandomForestClassifier(random_state = 101)

# Training the model:
rfc_mod = rfc.fit(f_train, t_train)

# Prediction
rfc_pred = rfc_mod.predict(f_test)


# In[25]:


print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, rfc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, rfc_pred)*100,2)))


# In[26]:


rfc = RandomForestClassifier(random_state = 101)
params = {
    'bootstrap': [True],
    'max_depth': [5, 8, 10, 15],
    'max_features': [2, 3, 4, 5, 6],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [10, 20, 30, 50]
}
grid_search = GridSearchCV(estimator=rfc, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=2, scoring = "accuracy")


# In[27]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(f_train, t_train)')


# In[28]:


grid_search.best_estimator_


# In[29]:


rfc = RandomForestClassifier(max_depth=15, max_features=6, min_samples_leaf=5,
                       min_samples_split=8, n_estimators=20, random_state=101)

# Training the model:
rfc_mod = rfc.fit(f_train, t_train)

# Prediction
rfc_pred = rfc_mod.predict(f_test)


# In[30]:


print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, rfc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, rfc_pred)*100,2)))


# #### The model performance improved after hyperparameter tuning
