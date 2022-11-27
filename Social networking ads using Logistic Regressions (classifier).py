#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df = pd.read_csv(r"C:\Users\Administrator\Downloads\social networking adds.csv")


# In[10]:


df.isna().sum()


# In[11]:


df.EstimatedSalary


# In[12]:


df.EstimatedSalary.median()


# In[26]:


df


# In[16]:


x=df.iloc[:,0:4].values
y=df.iloc[:,-1].values


# In[17]:


x


# In[18]:


y


# In[56]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.1,random_state=0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[27]:


x_train


# In[28]:


x_test


# In[ ]:


from sklearn.linear_model import LogisticRegression
classfier = LogisticRegression()
classfier.fit(x_train,y_train)


# In[ ]:


x_test=classfier.predict(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,x_test)
cm


# In[ ]:


from sklearn.metrics import accuracy_matrix
am = accuracy_matrix(y_test,x_test)
am

