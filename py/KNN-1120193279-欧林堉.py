#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
lilac_data = pd.read_csv('https://labfile.oss.aliyuncs.com/courses/1081/course-9-syringa.csv')
lilac_data.head()


# In[2]:


# !pip install -U scikit-learn


# In[3]:


import matplotlib.pyplot as plt
#绘制丁香花子图特征
fig,axes = plt.subplots(2,3,figsize=(20,10))
fig.subplots_adjust(hspace=0.3,wspace=0.2)

axes[0,0].set_xlabel("sepal_length")
axes[0,0].set_ylabel("sepal_width")
axes[0,0].scatter(lilac_data.sepal_length[:50],lilac_data.sepal_width[:50],c="b")
axes[0,0].scatter(lilac_data.sepal_length[50:100],lilac_data.sepal_width[50:100],c="g")
axes[0,0].scatter(lilac_data.sepal_length[100:],lilac_data.sepal_width[100:],c="r")
axes[0,0].legend(["daphne","syinga","willow"],loc=2)

axes[0,1].set_xlabel("sepal_length")
axes[0,1].set_ylabel("petal_length")
axes[0,1].scatter(lilac_data.sepal_length[:50],lilac_data.petal_length[:50],c="b")
axes[0,1].scatter(lilac_data.sepal_length[50:100],lilac_data.petal_length[50:100],c="g")
axes[0,1].scatter(lilac_data.sepal_length[100:],lilac_data.petal_length[100:],c="r")

axes[0,2].set_xlabel("sepal_length")
axes[0,2].set_ylabel("petal_width")
axes[0,2].scatter(lilac_data.sepal_length[:50],lilac_data.petal_width[:50],c="b")
axes[0,2].scatter(lilac_data.sepal_length[50:100],lilac_data.petal_width[50:100],c="g")
axes[0,2].scatter(lilac_data.sepal_length[100:],lilac_data.petal_width[100:],c="r")

axes[1,0].set_xlabel("sepal_width")
axes[1,0].set_ylabel("petal_width")
axes[1,0].scatter(lilac_data.sepal_width[:50],lilac_data.petal_width[:50],c="b")
axes[1,0].scatter(lilac_data.sepal_width[50:100],lilac_data.petal_width[50:100],c="g")
axes[1,0].scatter(lilac_data.sepal_width[100:],lilac_data.petal_width[100:],c="r")

axes[1,1].set_xlabel("sepal_width")
axes[1,1].set_ylabel("petal_length")
axes[1,1].scatter(lilac_data.sepal_width[:50],lilac_data.petal_length[:50],c="b")
axes[1,1].scatter(lilac_data.sepal_width[50:100],lilac_data.petal_length[50:100],c="g")
axes[1,1].scatter(lilac_data.sepal_width[100:],lilac_data.petal_length[100:],c="r")

axes[1,2].set_xlabel("petal_length")
axes[1,2].set_ylabel("petal_width")
axes[1,2].scatter(lilac_data.petal_length[:50],lilac_data.petal_width[:50],c="b")
axes[1,2].scatter(lilac_data.petal_length[50:100],lilac_data.petal_width[50:100],c="g")
axes[1,2].scatter(lilac_data.petal_length[100:],lilac_data.petal_width[100:],c="r")


# In[21]:


from sklearn.model_selection import train_test_split

# 得到 lilac 数据集中 feature 的全部序列: sepal_length,sepal_width,petal_length,petal_width
feature_data = lilac_data.iloc[:, :-1]
label_data = lilac_data["labels"]  # 得到 lilac 数据集中 label 的序列

X_train, X_test, y_train, y_test = train_test_split(
    feature_data, label_data, test_size=0.3, random_state=2)

np.array(X_test)  # 输出 lilac_test 查看


# In[24]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report

class Knn():
    #默认k=5，设置和sklearn中的一样
    def __init__(self):
        pass
    def fit(self,x,y,k=5):
        self.x = x
        self.y = y
        self.k = k
        
    def predict(self,x_test):
        labels = []
        #这里可以看出，KNN的计算复杂度很高，一个样本就是O(m * n)
        for i in range(len(x_test)):
            
            #初始化一个y标签的统计字典
            dict_y = {}
            #计算第i个测试数据到所有训练样本的欧氏距离
            diff = self.x - x_test[i]
            distances = np.sqrt(np.square(diff).sum(axis=1))
            
            #对距离排名，取最小的k个样本对应的y标签
            rank = np.argsort(distances)
            rank_k = rank[:self.k]
            y_labels = self.y[rank_k]
            
            #生成类别字典，key为类别，value为样本个数
            for j in y_labels:
                if j not in dict_y:
                    dict_y.setdefault(j,1)
                else:
                    dict_y[j] += 1
            
            #取得y_labels里面，value值最大对应的类别标签即为测试样本的预测标签
            
            #label = sorted(dict_y.items(),key = lambda x:x[1],reverse=True)[0][0]
            #下面这种实现方式更加优雅
            label = max(dict_y,key = dict_y.get)
            
            labels.append(label)
            
        return labels


# In[30]:


# 使用测试数据进行预测
my_classify = Knn()
my_classify.fit(np.array(X_train), np.array(y_train), 3)
y_predict = my_classify.predict(np.array(X_test))
print(y_predict)


# In[27]:


def ttt(y_predict):
    color_number = []
    for i in y_predict:
        if i == 'daphne':
            color_number.append(1)
        elif i == 'willow ':
            color_number.append(2)
        else:
            color_number.append(3)
    return color_number


# In[28]:


plt.scatter(X_test.sepal_length, X_test.sepal_width, marker='o',c=ttt(y_predict))


# In[29]:


def get_accuracy(test_labels, pred_labels):
    # 准确率计算函数
    correct = np.sum(test_labels == pred_labels)  # 计算预测正确的数据个数
    n = len(test_labels)  # 总测试集数据个数
    accur = correct/n
    return accur
get_accuracy(y_test, y_predict)


# In[14]:


normal_accuracy = []  # 建立一个空的准确率列表
k_value = range(2, 11)
for k in k_value:
    y_predict = sklearn_classify(X_train, y_train, X_test, k)
    accuracy = get_accuracy(y_test, y_predict)
    normal_accuracy.append(accuracy)

plt.xlabel("k")
plt.ylabel("accuracy")
new_ticks = np.linspace(0.6, 0.9, 10)  # 设定 y 轴显示，从 0.6 到 0.9
plt.yticks(new_ticks)
plt.plot(k_value, normal_accuracy, c='r')
plt.grid(True)  # 给画布增加网格


# In[ ]:





# In[ ]:




