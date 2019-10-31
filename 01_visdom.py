#!/usr/bin/env python
# coding: utf-8

# In[51]:


import torch# 라이브러리 불러오자
import torch.nn as nn

import torchvision
import torchvision.datasets as dsets


# In[52]:


# conda install -c conda-forge jsonpatch//설치
# conda install -c conda-forge visdom//서버를 강제로 켠다
# python -m visdom.server
import visdom# 
vis = visdom.Visdom()
# http://localhost:8097/ 로컬페이지


# In[53]:


vis.text("Hello, world!",env="main") #헬로 월드가 가상 창안에 뜬다


# In[54]:


a=torch.randn(3,200,200)#200*200 RGB 세 장을 볼수있따 랜덤으로 생성했고, 사실상 노이즈
vis.image(a)


# In[55]:


vis.images(torch.Tensor(3,3,28,28)) #디멘젼 다르게해서..


# In[56]:


# Download CIFAR10 dataset
cifar10 = dsets.CIFAR10(root="./cifar10",train = True, transform=torchvision.transforms.ToTensor(),download=True)# 이명령어가 하는일은 스탠다드 샛을 다운받아준다


# In[57]:


data = cifar10.__getitem__(0) #그림을 하나뽑아서 확인한다
print(data[0].shape)
vis.images(data[0],env="main") 


# In[58]:


data_loader = torch.utils.data.DataLoader(dataset = cifar10, #데이터를 넣으려고하는데 데이터가 너무크니까 쪼개서 넣는다.
                                          batch_size = 32,
                                          shuffle = False)


# for num, value in enumerate(data_loader):
#     value = value[0]
#     print(value.shape)
#     vis.images(value)
#     break

# In[59]:


vis.close(env="main")


# In[67]:


Y_data = torch.randn(5)
plt = vis.line (Y=Y_data)

X_data = torch.Tensor([1,2,3,4,5])
plt = vis.line(Y=Y_data, X=X_data)#그림 


# In[70]:


Y_append = torch.randn(1)
X_append = torch.Tensor([6])

vis.line(Y=Y_append, X=X_append, win=plt, update='append')#점을 추가하겠다


# In[69]:


num = torch.Tensor(list(range(0,10)))
num = num.view(-1,1)
num = torch.cat((num,num),dim=1)

plt = vis.line(Y=torch.randn(10,2), X = num)#y가 2차원 2*10짜리 그림, 길이 10짜리 선을 두개그리고싶음


# In[71]:


plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', showlegend=True))
plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', legend = ['1번'],showlegend=True))
plt = vis.line(Y=torch.randn(10,2), X = num, opts=dict(title='Test', legend=['1번','2번'],showlegend=True))


# In[72]:


def loss_tracker(loss_plot, loss_value, num):
    '''num, loss_value, are Tensor'''
    vis.line(X=num,
             Y=loss_value,
             win = loss_plot,
             update='append'
             )


# In[73]:


plt = vis.line(Y=torch.Tensor(1).zero_())

for i in range(500):
    loss = torch.randn(1) + i
    loss_tracker(plt, loss, torch.Tensor([i]))


# In[66]:


vis.close(env="main")


# In[ ]:




