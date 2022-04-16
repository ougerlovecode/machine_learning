#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy
import scipy
from sklearn.cluster import KMeans
import numpy as np
from itertools import cycle, islice


# In[7]:


from sklearn import datasets
from matplotlib import pyplot as plt
def genTwoCircles(n_samples=1000):
    X,y = datasets.make_circles(n_samples, factor=0.5, noise=0.05)
    return X, y
data, label = genTwoCircles(n_samples=500)
plt.scatter(data[:,0], data[:,1],s=10)


# In[8]:



def myKNN(S, k, sigma=1.0):
    N = len(S)
    #定义邻接矩阵
    A = np.zeros((N,N))
    for i in range(N):
        #对每个样本进行编号
        dist_with_index = zip(S[i], range(N))
        #对距离进行排序
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        #取得距离该样本前k个最小距离的编号
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours
        #构建邻接矩阵
        #高斯核函数RBF定义距离，最近的n个点
        for j in neighbours_id: # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
            A[j][i] = A[i][j] # mutually
    return A
def calLaplacianMatrix(adjacentMatrix):

    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

def euclidDistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res

def calEuclidDistanceMatrix(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
    return S


# In[14]:


np.random.seed(1)
data, label = genTwoCircles(n_samples=500)
data, label


# In[20]:


Similarity = calEuclidDistanceMatrix(data)
print(Similarity)


# In[19]:


Adjacent = myKNN(Similarity, k=10)
print(Adjacent)


# In[18]:


Laplacian = calLaplacianMatrix(Adjacent)
print(Laplacian)


# In[21]:


# 特征值分解
#计算Laplacian矩阵的特征值和右特征向量。
x, V = np.linalg.eig(Laplacian)
# 将特征向量按特征值大小，从小到大排列
x = zip(x, range(len(x)))
x = sorted(x, key=lambda x:x[0])
# 竖向堆叠
H = np.vstack([V[:,i] for (v, i) in x[:500]]).T
print(H)


# In[29]:


from sklearn.cluster import KMeans
def spKmeans(H):
    sp_kmeans = KMeans(n_clusters=2).fit(H)
    return sp_kmeans.labels_
labels = spKmeans(H)
plt.title('spectral cluster result')
plt.scatter(data[:, 0], data[:, 1], marker='o',c=labels)
plt.show()


# In[13]:


pure_kmeans = KMeans(n_clusters=2).fit(data)
plt.title('pure kmeans cluster result')
plt.scatter(data[:, 0], data[:, 1], marker='o',c=pure_kmeans.labels_)
plt.show()

