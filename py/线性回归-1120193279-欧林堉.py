#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
x = np.array([56,72,69,88,102,86,76,79,94,74])
y = np.array([92,102,86,110,130,99,96,102,105,92])


# In[2]:


from matplotlib import pyplot as plt

plt.scatter(x,y)
plt.xlabel('Area')
plt.ylabel('Price')


# In[3]:


def f(x,w0,w1):
    y = w0 + w1*x
    return y


# In[4]:


def square_loss(x,y,w0,w1):
    loss = sum(np.square(y-(w0+w1*x)))
    return loss


# In[5]:


def w_calculaor(x,y):
    n = len(x)
    w1 = (n*sum(x*y)- sum (x)*sum(y)) / (n* sum(x*x) - sum(x)*sum(x))
    w0 = (sum(x*x)*sum(y)- sum (x)*sum(x*y)) / (n* sum(x*x) - sum(x)*sum(x))
    return w0,w1


# In[6]:


w_calculaor(x,y)


# In[7]:


w0,w1 = w_calculaor(x,y)


# In[8]:


square_loss(x,y,w0,w1)


# In[9]:


x_temp = np.linspace(50,120,100)
plt.scatter(x,y)
plt.plot(x_temp,x_temp*w1 + w0,'r')


# In[59]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x.reshape(len(x),1),y)

model.intercept_,model.coef_


# In[62]:


def w_matrix(x,y):
    w = (x.T * x).I * x.T * y
    return w


# In[64]:


x = np.matrix([[1,56],[1,72],[1,69],[1,88],[1,102],
               [1,86],[1,76],[1,79],[1,94],[1,74]])
y = np.matrix([92,102,86,110,130,99,96,102,105,92])
w_matrix(x,y.reshape(10,1))


# In[30]:


import pandas as pd
df = pd.read_csv('C:/Users/14004/Desktop/boston.csv')


# In[31]:


df.head()


# In[42]:


features = df[['crim','rm','lstat']]
features.describe()


# In[45]:


target = df['medv']

split_num = int(len(features)*0.7)

X_train = features[:split_num]
Y_train = target[:split_num]

X_test = features[split_num:]
Y_test = target[split_num:]


# In[47]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)
model.coef_,model.intercept_


# In[48]:



preds = model.predict(X_test)
preds


# In[49]:


import numpy as np


# In[50]:


def mae_value(y_true,y_pred):
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred))/n
    return mae


# In[51]:


def mse_value(y_true,y_pred):
    n = len(y_true)
    mse = sum(np.square(y_true-y_pred))/n
    return mse


# In[53]:


mae = mae_value(Y_test.values,preds)
mse = mse_value(Y_test.values,preds)

print("MAE:",mae)
print("MSE:",mse)

