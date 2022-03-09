#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import warnings

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, classification_report
from sklearn.tree import plot_tree


# In[2]:


cwd = os.getcwd()


# In[3]:


ip_filepath = cwd+'\\WineQT.csv'


# In[4]:


wq_df = pd.read_csv(ip_filepath)


# In[5]:


print("Basic Profiling for the dataframe: ")
print("----------------------------------")
print("Number of features: ",wq_df.shape[1])
print("Number of records: ",wq_df.shape[0])
print("Size of the dataframe: ",wq_df.size,'')


# In[6]:


print("Detailed Statistics: ")
print("----------------------------------")
print(wq_df.describe())


# In[7]:


print("Detailed Information: ")
print("----------------------------------")
print(wq_df.info())


# In[8]:


print("Checking Null Values: ")
print(wq_df.isnull().sum())


# In[9]:


print(wq_df.dtypes.value_counts())
print("The dataframe has only numeric values")


# ### EDA

# In[10]:


wq = wq_df.copy()


# In[11]:


wq.head()


# In[12]:


wq.drop('Id',axis=1, inplace=True)
print("Dropped the Id column as it is an index column, and the df has already possess an index column")


# In[13]:


wq.head()


# #### The target variable is quality and all others are predictors

# ### Univariate Analysis:

# #### Target Variable - Quality

# In[14]:


def configure_plots(chart,xlab,ylab,desc):
    chart.set(xlabel = xlab, ylabel = ylab, title = desc)


# In[15]:


chart = sb.countplot(x='quality',data=wq)
configure_plots(chart,"WineQuality","Number of Wine Variaties","Frequency of wine variaties and its Quality")


# In[16]:


print(wq.columns)


# In[17]:


## Renaming the colmumns :- Space removal and Title Case

def replace_spaces(df):
    return df.columns.str.replace(" ","_")
def title_name(df):
    return df.columns.str.title()


# In[18]:


wq.columns = replace_spaces(wq)
wq.columns = title_name(wq)


# In[19]:


print(wq.columns)


# In[20]:


def feat_analysis(feature):
    
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    feature.plot(kind = 'hist')
    plt.title(f'{feature.name} histogram plot')
    
    plt.subplot(1, 3, 2)
    sb.distplot(feature)
    plt.title(f'{feature.name} distribution plot')
    
    plt.subplot(1, 3, 3)
    sb.boxplot(feature)
    plt.title(f'{feature.name} box plot')
    
    plt.show()


# In[21]:


warnings.filterwarnings("ignore")

for i in wq.columns[:11]:
    print(i.center(100,"*"))
    feat_analysis(wq[i])
    print()


# #### All the predictors are numeric and normally distributes

# ### Co-Relation between all the predictors vs target variable:

# In[22]:


plt.figure(figsize=(10,6))
sb.heatmap(wq.corr(),cmap="viridis", annot=True)
plt.show()


# In[23]:


target = wq.loc[:,'Quality']


# In[24]:


predictors = wq.loc[:, wq.columns != 'Quality']


# ### Splitting the data into train set and test set

# In[25]:


f_train, f_test, t_train, t_test = train_test_split(predictors, target, test_size = 0.3, random_state = 101)


# In[26]:


print(f'Shape of the f_train: {f_train.shape}')
print(f'Shape of the f_test: {f_test.shape}')
print(f'Shape of the t_train: {t_train.shape}')
print(f'Shape of the t_test: {t_test.shape}')


# ### Decision Tree Classifier

# In[27]:


dtc = DecisionTreeClassifier()
dtc_mod = dtc.fit(f_train, t_train)


# In[28]:


dtc_pred = dtc_mod.predict(f_test)


# In[29]:


print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))


# In[30]:


print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# ### Increased the size of train data

# In[31]:


f_train, f_test, t_train, t_test = train_test_split(predictors, target, test_size = 0.2, random_state = 101)


# In[32]:


dtc = DecisionTreeClassifier()
dtc_mod = dtc.fit(f_train, t_train)
dtc_pred = dtc_mod.predict(f_test)


# In[33]:


print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# #### The model performed well, Increase in accuracy score

# In[34]:


plt.figure(figsize=(15,12))
plot_tree(dtc, filled=True)
plt.show()


# #### Tuning the model for better performance

# In[35]:



dtc = DecisionTreeClassifier(min_samples_leaf = 5)
dtc_mod = dtc.fit(f_train, t_train)

# Prediction
dtc_pred = dtc_mod.predict(f_test)

# Model Evaluation using confusion Matrix and Accuracy Score
print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# In[36]:


plt.figure(figsize=(15,12))
plot_tree(dtc, filled=True)
plt.show()


# ### Hyperparameter tuning for multi class classification

# In[37]:


dtc = DecisionTreeClassifier(random_state=101)


# In[38]:


params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100]
}


# In[39]:


grid_search = GridSearchCV(estimator=dtc, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")


# In[40]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(f_train, t_train)')


# In[41]:


grid_search.best_estimator_


# In[42]:


dtc = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=101)
dtc_mod = dtc.fit(f_train, t_train)

# Prediction
dtc_pred = dtc_mod.predict(f_test)

# Model Evaluation using confusion Matrix and Accuracy Score
print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# In[43]:


plt.figure(figsize=(15,12))
plot_tree(dtc, filled=True)
plt.show()


# #### Let's transform the quality into binary values.
# 
# ##### 1. The wine quality with 3,4,5 shall be 'BAD' with value 0
# ##### 2. The wine quality with 6,7,8 shall be 'GOOD' with value 1

# In[44]:


target_binary = target.apply(lambda x: 1 if x>= 6 else 0)


# In[45]:


print(target_binary.value_counts())


# ### Let's Build the model again

# In[46]:


f_train, f_test, t_train, t_test = train_test_split(predictors, target_binary, test_size = 0.2, random_state = 101)

dtc = DecisionTreeClassifier(min_samples_leaf = 2)
dtc_mod = dtc.fit(f_train, t_train)


# In[47]:


# Prediction
dtc_pred = dtc_mod.predict(f_test)


# In[48]:


# Model Evaluation using confusion Matrix and Accuracy Score
print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# #### Get classification Report for Decsion Tree Classifier 

# In[49]:


target_names = ['BAD', 'GOOD']
print(classification_report(t_test, dtc_pred, target_names=target_names))


# ### Hyperparameter Tuning

# In[50]:


dtc = DecisionTreeClassifier(random_state=101)


# In[51]:


params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}


# In[52]:


grid_search = GridSearchCV(estimator=dtc, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")


# In[53]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(f_train, t_train)')


# In[54]:


grid_search.best_estimator_


# In[55]:



dtc = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=10,
                       random_state=101)
dtc_mod = dtc.fit(f_train, t_train)

# Prediction
dtc_pred = dtc_mod.predict(f_test)

# Model Evaluation using confusion Matrix and Accuracy Score
print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# #### After hyperparameter tuning the model performed well

# In[56]:


plt.figure(figsize=(10,10))
plot_tree(dtc, filled=True)
plt.show()

