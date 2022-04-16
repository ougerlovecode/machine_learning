#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#data
from sklearn import datasets
X,y = datasets.make_moons(n_samples=100,noise=0.005,random_state=666)


# In[2]:


plt.scatter(X[:,0],X[:,1],c=y)
plt.show()


# In[31]:


import numpy as np
import random
def findNeighbor(j,X,eps):
    N = []
    for p in range(X.shape[0]):
        temp=np.sqrt(np.sum(np.square(X[j]-X[p])))
        if(temp<=eps):
            N.append(p)
    return N
def dbscan(X,eps,min_Pts):
    k = -1
    NeighborPts = []
    Ner_NeighborPts = []
    fil = []
    gama = [x for x in range(len(X))]
    cluster = [-1 for y in range(len(X))]
    # 如果还有没访问的点
    while len(gama)>0:
        # 随机选择一个unvisited对象
        j = random.choice(gama)
        gama.remove(j)
        fil.append(j)
        # NeighborPts为p的epsilon邻域中的对象的集合
        NeighborPts = findNeighbor(j,X,eps)
        # 如果p的epsilon邻域中的对象数小于指定阈值，说明p是一个噪声点
        if len(NeighborPts) < min_Pts:
            cluster[j] = -1
        # 如果p的epsilon邻域中的对象数大于指定阈值，说明p是一个核心对象
        else:
            k = k+1
            cluster[j] = k
            # 对于p的epsilon邻域中的每个对象i
            for i in NeighborPts:
                if i not in fil:
                    gama.remove(i)
                    fil.append(i)
                    # 找到pi的邻域中的核心对象，将这些对象放入NeighborPts中
                    # Ner_NeighborPts是位于pi的邻域中的点的列表
                    Ner_NeighborPts = findNeighbor(i,X,eps)
                    if len(Ner_NeighborPts) >= min_Pts:
                        for a  in Ner_NeighborPts:
                            if a not in NeighborPts:
                                NeighborPts.append(a)
                    if (cluster[i] == -1):
                        cluster[i] = k
    return cluster


# In[41]:


y = np.array(y)
y


# In[48]:


y_pred = dbscan(X,eps=0.5,min_Pts=10)
y_pred = np.array(y_pred)
acc = accuracy_score(y,y_pred)
print("聚类的吻合度：{:.2f}%".format(acc * 100))
print(y_pred)


# In[49]:


y_pred[ y_pred==0 ]=2
y_pred[ y_pred==1 ]=0
y_pred[ y_pred==2 ]=1
acc = accuracy_score(y,y_pred)
print("聚类的吻合度：{:.2f}%".format(acc * 100))
print(y_pred)


# In[89]:


from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5,min_samples=10)
result = dbscan.fit_predict(X)
print(result)

result[ result==0 ]=2
result[ result==1 ]=0
result[ result==2 ]=1
acc = accuracy_score(y,result)
print("聚类的吻合度：{:.2f}%".format(acc * 100))


# In[51]:


plt.scatter(X[:,0],X[:,1],c=result)
plt.show()


# In[ ]:





# In[ ]:




