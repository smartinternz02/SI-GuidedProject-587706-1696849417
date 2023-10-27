#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# In[5]:


data = pd.read_csv("dataloss.csv")


# In[6]:


data.head()


# In[7]:


data.tail()


# In[9]:


data.describe()


# In[11]:


data.shape


# In[12]:


data.mean()


# In[13]:


data.median()


# In[14]:


data.mode()


# In[15]:


data.std()


# In[21]:


sns.displot(data['R&D Spend'])


# In[22]:


data.isnull()


# In[23]:


data.isnull().sum()


# In[24]:


sns.boxplot(data['Profit'])


# In[25]:


sns.boxplot(data['R&D Spend'])


# In[27]:


qnt = data.quantile(q =(0.25000,0.75000))
qnt


# In[28]:


iqr = qnt.loc[0.75000]-qnt.loc[0.25000]


# In[29]:


iqr


# In[31]:


upper = qnt.loc[0.75000]+1.5*iqr
upper


# In[33]:


lower = qnt.loc[0.25000]-1.5*iqr
lower


# In[34]:


data.mean()


# In[35]:


data.mode()


# In[36]:


data.median()


# In[42]:


data.head(2)


# In[46]:


data_main = pd.get_dummies(data,columns=['State'])


# In[47]:


data_main.head()


# In[48]:


y = data_main['Profit']


# In[51]:


x = data_main.drop(columns=['Profit'],axis=1)


# In[52]:


x.head()


# In[53]:


names = x.columns


# In[54]:


names


# In[55]:


from sklearn.preprocessing import scale


# In[56]:


x = scale(x)


# In[57]:


x


# In[58]:


x = pd.DataFrame(x,columns=names)


# In[59]:


x.head()


# In[61]:


from sklearn.model_selection import train_test_split


# In[63]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[64]:


x_train.head()


# In[65]:


x_test.head()


# In[66]:


y_train,y_test


# In[ ]:




