#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn import datasets


# In[3]:





# In[4]:

iris = datasets.load_iris() 
print(type(iris), dir(iris))

X = iris.get('data')
y = iris.get('target')

X = X[y<2,:2]
y = y[y<2]


plt.scatter(X[y==0,0],X[y==0,1],color='red')
plt.scatter(X[y==1,0],X[y==1,1],color='blue')


# In[54]:


from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
standardScaler.fit(X)
X_standerd = standardScaler.transform(X)
X =X_standerd


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)
#训练svm分类器，此时可以根据libsvm测得的C和gamma
clf = svm.LinearSVC(C=1e9)
clf.fit(X_standerd, y)
 


# In[56]:


def plot_svc_decision_boundary(model,axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),
        np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*100)).reshape(-1,1),
    )
    X_new = np.c_[x0.ravel(),x1.ravel()]

    y_predict = model.predict(X_new)
    zz= y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap =ListedColormap(['#EF9A9A','#FFF59D', '#90CAF9'])
    plt.contourf(x0,x1,zz,cmap=custom_cmap)
    
    #绘制上下两条决策边界线
    w=model.coef_[0]
    b = model.intercept_[0]

    plot_x = np.linspace(axis[0],axis[1],200)
    up_y = -w[0]/w[1] * plot_x - b/w[1] + 1/w[1]
    down_y = -w[0]/w[1] * plot_x - b/w[1] - 1/w[1]# wO*x0 + w1*x1+b=1
                                                    # wO*x0 +w1*x1 +b= -1
    up_index = (up_y >= axis[2]) & (up_y <= axis[3])#过滤up_y中超出轴范围的数
    down_index = (down_y >= axis[2]) & (down_y <= axis[3])#过滤up y中超出轴范围的数
    plt.plot(plot_x[up_index], up_y[up_index], color='black')#绘制两条边界线
    plt.plot(plot_x[down_index], down_y[down_index], color='black')
    
    


# In[57]:


plot_svc_decision_boundary(clf,axis=[-3,3,-3,3])
plt.scatter(X[y==0,0],X[y==0,1],color='red')
plt.scatter(X[y==1,0],X[y==1,1],color = 'blue')


# In[63]:


clf2 = svm.LinearSVC(C=1e-2)
clf2.fit(X_standerd, y)


# In[64]:


plot_svc_decision_boundary(clf2,axis=[-3,3,-3,3])
plt.scatter(X[y==0,0],X[y==0,1],color='red')
plt.scatter(X[y==1,0],X[y==1,1],color = 'blue')

