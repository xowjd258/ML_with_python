#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)

x_train = torch.FloatTensor(xy[:, 0:-1])
# squeeze : 2d->1d
y_train = torch.LongTensor(xy[:, [-1]]).squeeze()


# In[3]:


print(x_train[0, :], y_train[0])
print(x_train[1, :], y_train[1])
print(x_train[2, :], y_train[2])
print(x_train[3, :], y_train[3])


# In[4]:


# initialize
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 7)

    def forward(self, x):
        return self.linear(x)


# In[5]:


model = SoftmaxClassifierModel()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)


# In[6]:


nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # cost
    hypothesis = model(x_train)
    cost = F.cross_entropy(hypothesis, y_train)

    # gradient descent
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # check progress
    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f} ')


# In[7]:


params = list(model.parameters())
W = params[0]
b = params[1]
print(W, b)


# In[8]:


with torch.no_grad():
    output = model(x_train)
    prediction = torch.argmax(output, 1)
    print(prediction)
    
    correct_prediction = (prediction == y_train)
    print(correct_prediction)
    
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())


# In[ ]:





# In[ ]:




