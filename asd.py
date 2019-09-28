#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


torch.empty(5,3)


# In[3]:


x=torch.rand(5,3)


# In[4]:


print(x)


# In[5]:


torch.zeros(5,3,dtype=torch.long)


# In[6]:


a=[
    [1,2,3,4],[2,3,4,5]
]


# In[8]:


torch.FloatTensor(a)
x = torch.tensor(a)


# In[10]:


print(x.size())


# In[11]:


#행렬은 같은모양끼리만 더할 수 있다.
print(x+3)


# In[16]:


y=torch.tensor([[1],[2]])
print(x+y)
print(x*y)


# In[23]:


y[:,:]


# In[29]:


torch.cuda.is_available()


# In[ ]:




