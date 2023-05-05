#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


import numpy as np


# In[5]:


import seaborn as sea


# In[6]:


import matplotlib.pyplot as mat
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[9]:


data = pd.read_csv('IRIS.csv')


# In[10]:


data


# In[11]:


data.head()


# In[12]:


data.info()


# In[13]:


data.isnull().sum()


# In[14]:


data.columns


# In[15]:


data.describe()


# In[16]:


data['species'].value_counts()


# In[17]:


x=data.iloc[:,:4]
y=data.iloc[:,4]


# In[18]:


x


# In[19]:


y


# In[20]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[21]:


x_train.shape


# In[23]:


x_test.shape


# In[24]:


y_test.shape


# In[26]:


y_train.shape


# In[27]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[29]:


model.fit(x_train,y_train)


# In[30]:


y_pred=model.predict(x_test)


# In[31]:


y_pred


# In[33]:


from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_pred)


# In[35]:


accuracy=accuracy_score(y_test,y_pred)*100
print("Accuracy of the model is {:.2f}".format(accuracy))


# In[ ]:




