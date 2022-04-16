#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("C:/Users/14004/Desktop/course-6-vaccine.csv")


# In[2]:


df.head()


# In[3]:


import matplotlib.pyplot as plt
x= df['Year']
y = df["Values"]
plt.plot(x,y,'r')
plt.scatter(x,y)


# In[4]:


train_df = df[:int(len(df)*0.7)]
test_df = df[int(len(df)*0.7):]


# In[5]:


X_train = train_df['Year'].values
y_train = train_df['Values'].values
X_test = test_df['Year'].values
y_test = test_df['Values'].values


# In[6]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train.reshape(len(X_train),1),y_train.reshape(len(y_train),1))
results = model.predict(X_test.reshape(len(X_test),1))
results


# In[7]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print("线性回归平均绝对误差：",mean_absolute_error(y_test,results.flatten()))
print("线性回归均方误差：",mean_squared_error(y_test,results.flatten()))


# In[8]:


from sklearn.preprocessing import PolynomialFeatures

#二次多项式特征
poly_features_2 = PolynomialFeatures(degree = 2,include_bias=False)
poly_X_train_2 = poly_features_2.fit_transform(X_train.reshape(len(X_train),1))
poly_X_test_2 = poly_features_2.fit_transform(X_test.reshape(len(X_test),1))
model = LinearRegression()
model.fit(poly_X_train_2,y_train.reshape(len(X_train),1))
results_2 = model.predict(poly_X_test_2)
results_2.flatten()


# In[9]:


print("2次多项式线性回归平均绝对误差：",mean_absolute_error(y_test,results_2.flatten()))
print("2次多项式线性回归均方误差：",mean_squared_error(y_test,results_2.flatten()))


# In[14]:


from sklearn.pipeline import make_pipeline
X_train = X_train.reshape(len(X_train),1)
X_test = X_test.reshape(len(X_test),1)
y_train = y_train.reshape(len(y_train),1)
for m in [3,4,5,6,7,8,9,10,11,12]:
    model = make_pipeline(PolynomialFeatures(m,include_bias=False),LinearRegression())
    model.fit(X_train,y_train)
    pre_y = model.predict(X_test)
    print("{}次多项式线性回归平均绝对误差：".format(m),mean_absolute_error(y_test,pre_y.flatten()))
    print("{}次多项式线性回归均方误差：".format(m),mean_squared_error(y_test,pre_y.flatten()))


# In[17]:


mse = []
m = 1
m_max = 20
while m <= m_max:
    model = make_pipeline(PolynomialFeatures(m,include_bias=False),LinearRegression())
    model.fit(X_train,y_train)
    pred_y = model.predict(X_test)
    mse.append(mean_squared_error(y_test,pred_y.flatten()))
    m=m+1
plt.plot([i for i in range(1,m_max + 1)],mse,'r')
plt.scatter([i for i in range(1,m_max + 1)],mse)
plt.title("MSE of m degree of polynomial regression")
plt.xlabel('m')
plt.ylabel('MSE')


# In[ ]:





# In[ ]:




