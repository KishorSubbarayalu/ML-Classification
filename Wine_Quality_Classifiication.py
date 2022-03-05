#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


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


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sb


# #### The target variable is quality and all others are predictors

# ### Univariate Analysis:

# #### Target Variable - Quality

# In[15]:


def configure_plots(chart,xlab,ylab,desc):
    chart.set(xlabel = xlab, ylabel = ylab, title = desc)


# In[16]:


chart = sb.countplot(x='quality',data=wq)
configure_plots(chart,"WineQuality","Number of Wine Variaties","Frequency of wine variaties and its Quality")


# In[17]:


print(wq.columns)


# In[18]:


## Renaming the colmumns :- Space removal and Title Case

def replace_spaces(df):
    return df.columns.str.replace(" ","_")
def title_name(df):
    return df.columns.str.title()


# In[19]:


wq.columns = replace_spaces(wq)
wq.columns = title_name(wq)


# In[20]:


print(wq.columns)


# In[21]:


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


# In[22]:


import warnings
warnings.filterwarnings("ignore")

for i in wq.columns[:11]:
    print(i.center(100,"*"))
    feat_analysis(wq[i])
    print()


# #### All the predictors are numeric and normally distributes

# ### Co-Relation between all the predictors vs target variable:

# In[23]:


plt.figure(figsize=(10,6))
sb.heatmap(wq.corr(),cmap="viridis", annot=True)
plt.show()


# In[24]:


print(wq.columns)


# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[26]:


target = wq.loc[:,'Quality']


# In[27]:


predictors = wq.loc[:, wq.columns != 'Quality']


# ### Splitting the data into train set and test set

# In[28]:


f_train, f_test, t_train, t_test = train_test_split(predictors, target, test_size = 0.3, random_state = 101)


# In[29]:


print(f'Shape of the f_train: {f_train.shape}')
print(f'Shape of the f_test: {f_test.shape}')
print(f'Shape of the t_train: {t_train.shape}')
print(f'Shape of the t_test: {t_test.shape}')


# ### Decision Tree Classifier

# In[30]:


dtc = DecisionTreeClassifier()
dtc_mod = dtc.fit(f_train, t_train)


# In[31]:


dtc_pred = dtc_mod.predict(f_test)


# In[32]:


print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))


# In[33]:


print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# ### Increased the size of train data, the model performed well

# In[34]:


f_train, f_test, t_train, t_test = train_test_split(predictors, target, test_size = 0.2, random_state = 101)


# In[35]:


dtc = DecisionTreeClassifier()
dtc_mod = dtc.fit(f_train, t_train)
dtc_pred = dtc_mod.predict(f_test)


# In[36]:


print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# ### Standardize and fiting the model

# In[37]:


from sklearn.preprocessing import StandardScaler


# In[38]:


scaler = StandardScaler()


# In[39]:


f_train, f_test, t_train, t_test = train_test_split(predictors, target, test_size = 0.3, random_state = 101)


# In[40]:


scaler.fit(f_train)


# In[41]:


print("Before Standardizing: \n {} ".format(f_train.head()))


# In[42]:


f_train=(scaler.transform(f_train))
print("After Standardizing: \n {} ".format(f_train))


# In[43]:


# Similarly, modify the scales in test dataset
scaler.fit(f_test)
f_test=(scaler.transform(f_test))


# In[44]:


# Fit the model
dtc = DecisionTreeClassifier()
dtc_mod = dtc.fit(f_train, t_train)


# In[45]:


# Predict the wine quality
dtc_pred = dtc_mod.predict(f_test)


# In[46]:


print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# In[47]:


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

# In[48]:


scaler.fit(predictors)
predictors=(scaler.transform(predictors))


# In[49]:


f_train, f_test, t_train, t_test = train_test_split(predictors, target, test_size = 0.3, random_state = 101)

dtc = DecisionTreeClassifier()
dtc_mod = dtc.fit(f_train, t_train)

# Prediction
dtc_pred = dtc_mod.predict(f_test)

# Model Evaluation using confusion Matrix and Accuracy Score
print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# In[50]:


f_train, f_test, t_train, t_test = train_test_split(predictors, target, test_size = 0.2, random_state = 101)

dtc = DecisionTreeClassifier()
dtc_mod = dtc.fit(f_train, t_train)

# Prediction
dtc_pred = dtc_mod.predict(f_test)

# Model Evaluation using confusion Matrix and Accuracy Score
print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# In[51]:


from sklearn.tree import plot_tree


# In[52]:


plot_tree(dtc)


# #### Tuning the model for better performance

# In[53]:



dtc = DecisionTreeClassifier(min_samples_leaf = 5)
dtc_mod = dtc.fit(f_train, t_train)

# Prediction
dtc_pred = dtc_mod.predict(f_test)

# Model Evaluation using confusion Matrix and Accuracy Score
print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# In[54]:


plt.figure(figsize=(15,12))
plot_tree(dtc, filled=True)
plt.show()


# #### Let's transform the quality into binary values.
# 
# ##### 1. The wine quality with 3,4,5 shall be 'BAD' with value 0
# ##### 2. The wine quality with 6,7,8 shall be 'GOOD' with value 1

# In[92]:


target_binary = target.apply(lambda x: 1 if x>= 6 else 0)


# In[93]:


print(target_binary.value_counts())


# ### Let's Build the model again

# In[94]:


f_train, f_test, t_train, t_test = train_test_split(predictors, target_binary, test_size = 0.2, random_state = 101)

dtc = DecisionTreeClassifier(min_samples_leaf = 2)
dtc_mod = dtc.fit(f_train, t_train)


# In[95]:


# Prediction
dtc_pred = dtc_mod.predict(f_test)


# In[96]:


# Model Evaluation using confusion Matrix and Accuracy Score
print("Confusion Matrix of the Decision Tree Model: \n {}".format(confusion_matrix(t_test, dtc_pred)))
print("Accuracy score of the Decision Tree Model: \n{} %".format(round(accuracy_score(t_test, dtc_pred)*100,2)))


# #### Get classification Report for Decsion Tree Classifier 

# In[103]:


from sklearn.metrics import classification_report


# In[104]:


target_names = ['BAD', 'GOOD']
print(classification_report(t_test, dtc_pred, target_names=target_names))

