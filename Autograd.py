#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch 


# In[14]:


x=torch.ones(2,2, requires_grad=True) #requires_grad=TRUE는 역추적 시작을 나타냄!
x


# In[15]:


y= x+2
y


# In[16]:


z= y*y*3
z


# In[17]:


z.mean()


# In[18]:


out=z.mean()#평균을 나타내는 .mean
out


# In[19]:


out.backward() #역추적을 나타내는 backward


# In[20]:


print(x.grad) #??? 1과 4.5의 상관관계..? grad는 기울기를 나타낸다!


# In[24]:


a= torch. randn(3, requires_grad=True)
b=a*2
while b.data.norm()<1000:
    b=b*2
print(b)


# In[ ]:





# In[ ]:




