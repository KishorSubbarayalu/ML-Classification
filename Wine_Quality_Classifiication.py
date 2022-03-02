#!/usr/bin/env python
# coding: utf-8

# In[325]:


import pandas as pd
import numpy as np
import os


# In[326]:


cwd = os.getcwd()


# In[327]:


ip_filepath = cwd+'\\WineQT.csv'


# In[328]:


wq_df = pd.read_csv(ip_filepath)


# In[329]:


print("Basic Profiling for the dataframe: ")
print("----------------------------------")
print("Number of features: ",wq_df.shape[1])
print("Number of records: ",wq_df.shape[0])
print("Size of the dataframe: ",wq_df.size,'')


# In[330]:


print("Detailed Statistics: ")
print("----------------------------------")
print(wq_df.describe())


# In[331]:


print("Detailed Information: ")
print("----------------------------------")
print(wq_df.info())


# In[332]:


print("Checking Null Values: ")
print(wq_df.isnull().sum())


# In[333]:


print(wq_df.dtypes.value_counts())
print("The dataframe has only numeric values")


# ### EDA

# In[334]:


wq = wq_df.copy()


# In[335]:


wq.head()


# In[336]:


wq.drop('Id',axis=1, inplace=True)
print("Dropped the Id column as it is an index column, and the df has already possess an index column")


# In[337]:


wq.head()


# In[338]:


import matplotlib.pyplot as plt
import seaborn as sb


# #### The target variable is quality and all others are predictors

# ### Univariate Analysis:

# #### Target Variable - Quality

# In[339]:


def configure_plots(chart,xlab,ylab,desc):
    chart.set(xlabel = xlab, ylabel = ylab, title = desc)


# In[340]:


chart = sb.countplot(x='quality',data=wq)
configure_plots(chart,"WineQuality","Number of Wine Variaties","Frequency of wine variaties and its Quality")


# In[342]:


print(wq.columns)


# In[343]:


## Renaming the colmumns :- Space removal and Title Case

def replace_spaces(df):
    return df.columns.str.replace(" ","_")
def title_name(df):
    return df.columns.str.title()


# In[344]:


wq.columns = replace_spaces(wq)
wq.columns = title_name(wq)


# In[345]:


print(wq.columns)


# In[346]:


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


# In[347]:


import warnings
warnings.filterwarnings("ignore")

for i in wq.columns[:11]:
    print(i.center(100,"*"))
    feat_analysis(wq[i])
    print()


# #### All the predictors are numeric and normally distributes

# ### Co-Relation between all the predictors vs target variable:

# In[348]:


plt.figure(figsize=(10,6))
sb.heatmap(wq.corr(),cmap="viridis", annot=True)
plt.show()


# In[349]:


print(wq.columns)


# In[350]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[351]:


target = wq.loc[:,'Quality']


# In[352]:


predictors = wq.loc[:, wq.columns != 'Quality']


# ### Splitting the data into train set and test set

# In[353]:


f_train, f_test, t_train, t_test = train_test_split(predictors, target, test_size = 0.3, random_state = 101)


# In[354]:


print(f'Shape of the f_train: {f_train.shape}')
print(f'Shape of the f_test: {f_test.shape}')
print(f'Shape of the t_train: {t_train.shape}')
print(f'Shape of the t_test: {t_test.shape}')


# ### Decision Tree Classifier

# In[355]:


dtc = DecisionTreeClassifier()
dtc_mod = dtc.fit(f_train, t_train)


# In[356]:


dtc_pred = dtc_mod.predict(f_test)


# In[357]:


print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))


# In[358]:


print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# ### Increased the size of train data, the model performed well

# In[359]:


f_train, f_test, t_train, t_test = train_test_split(predictors, target, test_size = 0.2, random_state = 101)


# In[360]:


dtc = DecisionTreeClassifier()
dtc_mod = dtc.fit(f_train, t_train)
dtc_pred = dtc_mod.predict(f_test)


# In[361]:


print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# ### Standardize and fiting the model

# In[362]:


from sklearn.preprocessing import StandardScaler


# In[363]:


scaler = StandardScaler()


# In[364]:


f_train, f_test, t_train, t_test = train_test_split(predictors, target, test_size = 0.3, random_state = 101)


# In[365]:


scaler.fit(f_train)


# In[366]:


print("Before Standardizing: \n {} ".format(f_train.head()))


# In[367]:


f_train=(scaler.transform(f_train))
print("After Standardizing: \n {} ".format(f_train))


# In[368]:


# Similarly, modify the scales in test dataset
scaler.fit(f_test)
f_test=(scaler.transform(f_test))


# In[369]:


# Fit the model
dtc = DecisionTreeClassifier()
dtc_mod = dtc.fit(f_train, t_train)


# In[370]:


# Predict the wine quality
dtc_pred = dtc_mod.predict(f_test)


# In[371]:


print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# In[372]:


# Changed the train-test split size:
f_train, f_test, t_train, t_test = train_test_split(predictors, target, test_size = 0.2, random_state = 101)

# Standardize the train data
scaler.fit(f_train)
f_train=(scaler.transform(f_train))

# Standardize the test data
scaler.fit(f_test)
f_test=(scaler.transform(f_test))

# Fitting the model
dtc = DecisionTreeClassifier()
dtc_mod = dtc.fit(f_train, t_train)

# Prediction
dtc_pred = dtc_mod.predict(f_test)

# Model Evaluation using confusion Matrix and Accuracy Score
print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# ### Standardize Whole Predictors and then performing train-test split

# In[373]:


scaler.fit(predictors)
predictors=(scaler.transform(predictors))


# In[374]:


f_train, f_test, t_train, t_test = train_test_split(predictors, target, test_size = 0.3, random_state = 101)

dtc = DecisionTreeClassifier()
dtc_mod = dtc.fit(f_train, t_train)

# Prediction
dtc_pred = dtc_mod.predict(f_test)

# Model Evaluation using confusion Matrix and Accuracy Score
print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# In[375]:


f_train, f_test, t_train, t_test = train_test_split(predictors, target, test_size = 0.2, random_state = 101)

dtc = DecisionTreeClassifier()
dtc_mod = dtc.fit(f_train, t_train)

# Prediction
dtc_pred = dtc_mod.predict(f_test)

# Model Evaluation using confusion Matrix and Accuracy Score
print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))
