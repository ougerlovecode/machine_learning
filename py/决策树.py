#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np # linear algebra


# In[2]:


data = pd.read_csv('student-por.csv')


# In[3]:


data.head()


# In[4]:


# Create target object and call it y
y = data.G3
# Create X
X = data.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,14,15,30,31,32]]
X.head()


# In[7]:


def chioce2(x):
    x= int (x)
    if x<5:
        return 'bad'
    elif x>=5 and x<10:
        return 'medium'
    elif x>=10 and x<15:
        return 'good'
    else:
        return 'excellent'


# In[8]:


data = X.copy()
data['G1'] = pd.Series(map(lambda x: chioce2(x), data['G1']))
data['G2'] = pd.Series(map(lambda x: chioce2(x), data['G2']))
data['G3'] = pd.Series(map(lambda x: chioce2(x), data['G3']))


# In[9]:


data.head()


# In[10]:


def choice_3(x):
    x = int(x)
    if x> 3:
        return 'high'
    elif x > 1.5:
        return 'medium'
    else:
        return 'low'


# In[11]:


data['Medu'] = pd.Series(map(lambda x: choice_3(x),data['Medu']))
data.head()


# In[23]:


def replace_feature(data):
    for each in data.columns:
        feature_list = data[each]
        unique_v = set(feature_list)
        i= 0
        for fe_value in unique_v:
            data[each] = data[each].replace(fe_value,i)
            i += 1
    return data


# In[24]:


data  = replace_feature(data)
data.head()


# In[25]:


X_train,X_test,y_train,y_test = train_test_split(data.iloc[:,:-1],data['G3'],
                                                test_size=0.3,random_state=5)
X_test.head()


# In[30]:


dt_model = DecisionTreeClassifier(criterion='entropy',random_state=666)
dt_model.fit(X_train,y_train)


# In[31]:


y_pred = dt_model.predict(X_test)
y_pred


# In[32]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

