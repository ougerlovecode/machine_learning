#!/usr/bin/env python
# coding: utf-8

# In[1]:



from mindspore import context

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


# In[2]:


# 用梯度下降法求 y = (x-2.5)**2- 1 的极值点


# In[3]:


import numpy as np


# In[4]:


x = np.linspace(-1,6,141)
y = (x-2.5)**2- 1


# In[10]:


import matplotlib.pyplot as plt


# In[12]:


plt.plot(x,y)


# In[15]:


epsilon = 1e-8
eta = 0.1


# In[16]:


def J(theta):
    return (theta-2.5)**2-1
def dJ(theta):
    return 2*(theta-2.5)


# In[17]:


# theta_history = []
# def gradient_descent(initial_theta,eta,epsilon = 1e-8):
#     theta = initial_theta
#     theta_history.append(initial_theta)
#     while True:
#         gradient = dJ(theta)
#         last_theta = theta
#         theta = theta - eta * gradient
#         theta_history.append(theta)
        
#         if (abs(J(theta)-J(last_theta)) < epsilon):
#             break
        


# In[26]:


theta_history = []
def gradient_descent(initial_theta,eta,n_iters=1e4,epsilon = 1e-8):
    theta = initial_theta
    i_iter = 0
    theta_history.append(initial_theta)
    while i_iter < n_iters:
        gradient = dJ(theta)
        last_theta = theta
        theta = theta - eta * gradient
        theta_history.append(theta)
        if (abs(J(theta)-J(last_theta)) < epsilon):
            break
        i_iter += 1
    return


# In[21]:


# 封装绘制梯度下降过程
def plot_tehta_history():
    plt.plot(x,(y))#绘制函数曲线
    # 绘制梯度下降过程
    plt.plot(np.array(theta_history)
             ,J(np.array(theta_history)),
             color='r',marker = '+')


# In[22]:


eta = 0.01
theta_history = []
gradient_descent(0,eta)
plot_tehta_history()


# In[24]:


eta = 0.8
theta_history = []
gradient_descent(0,eta)
plot_tehta_history()


# In[30]:


eta = 1.1
theta_history = []
gradient_descent(0,eta,n_iters=10)
plot_tehta_history()


# ## 最小二乘法

# In[36]:


x = np.array([55,71,68,87,101,87,75,78,93,73])
y = np.array([91,101,87,109,129,98,95,101,104,93])


# In[34]:


def ols_algebra(x,y):
    '''
    x -- 自变量
    y -- 因变量
    
    返回
    w1 -- 线性方程系数
    w0 -- 线性方程的截距
    '''
    n = len(x)
    w1 = (n* sum(x*y) - sum(x)*sum(y)) / (n*sum(x*x) - sum(x)*sum(x))
    w0 = (sum(x*x)*sum(y) - sum(x)*sum(x*y)) / (n*sum(x*x) - sum(x)*sum(x))
    
    return w1,w0


# In[38]:


def ols_gradient_descent(x,y,lr,num_iter):
    '''
    x -- 自变量
    y -- 因变量
    
    返回
    w1 -- 线性方程系数
    w0 -- 线性方程的截距
    '''
    w1 = 0
    w0 = 0
    for i in range(num_iter):
        y_hat = (w1*x)+w0
        w1_gradient = -2 * sum(x*(y-y_hat))
        w0_gradient = -2 * sum(y-y_hat)
        w1 -= lr * w1_gradient
        w0 -= lr * w0_gradient
    return w1 ,w0


# In[50]:


w1 ,w0 = ols_algebra(x,y)
print(w1)
print(w0)


# In[54]:


w1_ ,w0_ = ols_gradient_descent(x,y,lr = 1e-5,num_iter=500)
print(w1_)
print(w0_)


# In[51]:


fig ,axes = plt.subplots(1,2,figsize=(15,5))
axes[0].scatter(x,y)
axes[0].plot(np.array([50,110]),np.array([50,110]) * w1 + w0,'r')
axes[0].set_title("OLS")

axes[1].scatter(x,y)
axes[1].plot(np.array([50,110]),np.array([50,110]) * w1_ + w0_,'r')
axes[1].set_title("Gradient descent")


# In[56]:


w1_ ,w0_ = ols_gradient_descent(x,y,lr = 1e-5,num_iter=1200000)
print(w1_)
print(w0_)


# In[57]:


fig ,axes = plt.subplots(1,2,figsize=(15,5))
axes[0].scatter(x,y)
axes[0].plot(np.array([50,110]),np.array([50,110]) * w1 + w0,'r')
axes[0].set_title("OLS")

axes[1].scatter(x,y)
axes[1].plot(np.array([50,110]),np.array([50,110]) * w1_ + w0_,'r')
axes[1].set_title("Gradient descent")

