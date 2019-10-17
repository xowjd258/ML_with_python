#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32) #컬럼갯수가 16개

x_train = torch.FloatTensor(xy[:, 0:-1]) 
# squeeze : 2d->1d
y_train = torch.LongTensor(xy[:, [-1]]).squeeze() #정답을 포함하는 y값


# In[3]:


print(x_train[0, :], y_train[0]) #input 0 표현, 카테고리 0
print(x_train[1, :], y_train[1])
print(x_train[2, :], y_train[2])#카테고리 3표현
print(x_train[3, :], y_train[3])

#이런 데이터를 많이보고 판단할 수 있어야한다
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

# optimizer, 오차값이 최소화 하도록 만드는 방식을 찾는다
optimizer = optim.SGD(model.parameters(), lr=0.01) #SGD방식을 통해 모델에있는 파라메터들을 알고 싶다. 러닝레이트가 작으면 최소값을 구하는데 오래걸림


# In[6]:


nb_epochs = 1000 #몇번 스탭을 갈꺼냐 1000번 밟겠다, epoch이 부족하면 러닝레이트가 짧아지더라도 정확도가 
for epoch in range(nb_epochs + 1): #사실상 1001회 회전

    # cost
    hypothesis = model(x_train)
    cost = F.cross_entropy(hypothesis, y_train)#코스트가 작으면 작으면 작을 수록 좋다

    # gradient descent
    optimizer.zero_grad()#그래디언트 초기화
    cost.backward()#그래디언트 계산
    optimizer.step()#그래디언트 쪽으로 움직임

    # check progress
    if epoch % 100 == 0: #100번마다 한번 코스트와 진행도를 볼 수 있다
        print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f} ')


# In[7]:


params = list(model.parameters())
W = params[0] #a 파라미터 갯수 16*7
b = params[1] #7
print(W, b)


# In[8]:


with torch.no_grad(): #테스트를 하는 코드,모델은 픽스한채로 내 값과 얼마나 일치하는지
    output = model(x_train) 
    prediction = torch.argmax(output, 1) #확률이 제일 높은친구가 나의 guess가 될 것이다
    print(prediction) #예측
    
    correct_prediction = (prediction == y_train) #실제 값
    print(correct_prediction)
    
    accuracy = correct_prediction.float().mean() #트루의 비율이 나의 정답률이다
    print('Accuracy:', accuracy.item())

