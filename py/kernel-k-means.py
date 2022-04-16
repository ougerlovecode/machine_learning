#!/usr/bin/env python
# coding: utf-8

# In[13]:


from math import exp 

def kernel(data, sigma):
    """
    RBF kernel-k-means
    :param data: data points: list of list [[a,b],[c,d]....]
    :param sigma: Gaussian radial basis function
    :return:
    """
    nData = len(data)
    Gram = [[0] * nData for i in range(nData)] # nData x nData matrix
    # TODO
    # Calculate the Gram matrix

    # symmetric matrix
    for i in range(nData):
        for j in range(i,nData):
            if i != j: # diagonal element of matrix = 0
                # RBF kernel: K(xi,xj) = e ( (-|xi-xj|**2) / (2sigma**2)
                square_dist = squaredDistance(data[i],data[j])
                base = 2.0 * sigma**2
                Gram[i][j] = exp(-square_dist/base)
                Gram[j][i] = Gram[i][j]
    return Gram 


# In[5]:


def loadPoints(filename):
    input = open(filename, "r")
    
    info = input.readline().split()
    
# number of data points and dimension
    # already know: (1)# of data points (2)dimension --> first line of the data file
    nData = int(info[0]) 
    nDim = int(info[1])
    
# create data matrix
    data = [[0]*nDim for i in range(nData)]

    for i in range(nData):
        info = input.readline().split()
        for j in range(nDim):
            data[i][j] = float(info[j]) 

    return data 


# In[7]:


def loadClusters(filename): 
    input = open(filename, "r") 
    
    info = input.readline() 
    
    nData = int(info)
    
    clusters = [0] * nData 
    
    for i in range(nData):
        info = input.readline()
        clusters[i] = int(info)
    
    return clusters


# In[15]:


def squaredDistance(vec1, vec2):
    sum = 0 
    dim = len(vec1) 
    
    for i in range(dim):
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]) 
    
    return sum


# In[20]:


def updateClusterID(data, centers):
    """
    assign the closet center to each data point
    :param data: data points: list of list [[a,b],[c,d]....]
    :param centers: data points: list of list [[a,b],[c,d]], K=2 (according to main)
    :return:
    """
    nData = len(data) # how many data points
    nCenters = len(centers) # how many centers

    
    clusterID = [0] * nData
    dis_Centers = [0] * nCenters# the distance between one data point to each center, since K=2, list [len1,len2]
    
    # TODO 
    # assign the closet center to each data point
    for i in range(nData):
        for c in range(nCenters):
            # calculate the distance between one data point to one center
            dis_Centers[c] = squaredDistance(data[i], centers[c])
        # assign the closet center to the data point, clusterID: the index of the dis_Centers list
        clusterID[i] = dis_Centers.index(min(dis_Centers))
    return clusterID


# In[22]:


# K: number of clusters 
def updateCenters(data, clusterID, K):
    nDim = len(data[0]) # the dimension of one data point
    centers = [[0] * nDim for i in range(K)] # list of list [[a,b],[c,d]]

    # TODO recompute the centers based on current clustering assignment
    # If a cluster doesn't have any data points, in this homework, leave it to ALL 0s
    ids = sorted(set(clusterID)) # sorted unique clusterID
    for id in ids:
        # get the index from clusterID where data points belong to the same cluster
        indices = [i for i, j in enumerate(clusterID) if j == id]
        # all data point in the same cluster, list of lists [[a,b],[c,d]...]
        cluster = [data[i] for i in indices]
        if len(cluster) == 0:
            #If a cluster doesn't have any data points, leave it to ALL 0s
            centers[id] = [0] * nDim
        else:
            # compute the centroids (i.e., mean point) of each cluster
            centers[id] = [float(sum(col))/len(col) for col in zip(*cluster)]
    return centers 


# In[25]:


def computeSSE(data, centers, clusterID):
    """
    objective function: calculate Sum of Squared Errors
    :param data:
    :param centers:
    :param clusterID:
    :return:
    """
    sse = 0 
    nData = len(data) 
    for i in range(nData):
        c = clusterID[i]
        sse += squaredDistance(data[i], centers[c]) 
        
    return sse 


# In[18]:


def kmeans(data, centers, maxIter = 100, tol = 1e-6):
    """
    :param data: data points: list of list [[a,b],[c,d]....]
    :param centers: data points: list of list [[a,b],[c,d]], K=2 (according to main)
    :param maxIter:
    :param tol:
    :return: clusterID: list
    """
    nData = len(data) 
    
    if nData == 0:
        return []

    K = len(centers) 
    
    clusterID = [0] * nData
    
    if K >= nData:
        for i in range(nData):
            clusterID[i] = i
        return clusterID

    nDim = len(data[0]) 
    
    lastDistance = 1e100
    
    for iter in range(maxIter):
        clusterID = updateClusterID(data, centers) 
        centers = updateCenters(data, clusterID, K)
        
        curDistance = computeSSE(data, centers, clusterID) # objective function
        if lastDistance - curDistance < tol or (lastDistance - curDistance)/lastDistance < tol:
#             print "# of iterations:", iter 
#             print "SSE = ", curDistance
            return clusterID
        
        lastDistance = curDistance
        
#     print "# of iterations:", iter 
#     print "SSE = ", curDistance
    return clusterID


# In[88]:


dataFilename = 'data/quiz_C.data'
groundtruthFilename = 'data/quiz_C.ground'

data = loadPoints(dataFilename) 
groundtruth = loadClusters(groundtruthFilename) 
sigma = 4.0
    
data2 = kernel(data, sigma)  

nDim = len(data[0]) 

K = 2  # Suppose there are 2 clusters
print ('K=',K)

centers = []
for i in range(K):
    centers.append(data2[i])

results = kmeans(data2, centers)


# In[89]:


x,y = np.array(data)[:,0],np.array(data)[:,1]
import matplotlib.pyplot as plt
plt.scatter(x,y)


# In[90]:


x,y = np.array(data)[:,0],np.array(data)[:,1]
import matplotlib.pyplot as plt
plt.scatter(x,y,color='r')
x,y  = np.array(data)[:,0] * np.array(groundtruth) , np.array(data)[:,1] * np.array(groundtruth)
plt.scatter(x,y)


# In[91]:


x,y = np.array(data)[:,0],np.array(data)[:,1]
import matplotlib.pyplot as plt
plt.scatter(x,y,color='r')
x,y  = np.array(data)[:,0] * np.array(results) , np.array(data)[:,1] * np.array(results)
plt.scatter(x,y)

