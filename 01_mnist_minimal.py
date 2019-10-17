#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random


# In[10]:


# parameters
training_epochs = 15
batch_size = 100

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(), #텐서형태로 데이터를 바꾸고 다운받아라
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', #위에는 true경우 아래는 false경우
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, #학습할때마다
                                          batch_size=batch_size, 
                                          shuffle=True,#혼선이 생길수 있으니 epoch주기마다  셔플한다
                                          drop_last=True)# 데이터가 230개있으면 30개를 버린다


# In[21]:


r = random.randint(0, len(mnist_test) - 1)#r이라는 테스트 값을 뽑아서 레이블을 테스트 하겠다
X_single_data = mnist_test.test_data[r:r + 1].view(28, 28)
Y_single_data = mnist_test.test_labels[r:r + 1]

print('Label: ', Y_single_data.item())
plt.imshow(X_single_data, cmap='Greys', interpolation='nearest')
plt.show()


# In[17]:


# MNIST data image of shape 28 * 28 = 784
linear = torch.nn.Linear(784, 10, bias=True) #입력이 784개 카테고리 10 bias는 B값 쓸껀지 안쓸껀지

# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss()    # Softmax is internally computed. 로스함수
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)#옵티마이저 함수, 파라메타들을 학습하겠다.


# In[12]:


for epoch in range(training_epochs): #0부터 14까지 총 15회 돌릴꺼다
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader: #데이터로더는 처음 100개의 레이블과 이미지 셋을 가져온거임
        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        X = X.view(-1, 28 * 28)#y축의 갯수를 768개로 만들고 x축은 니 알아서해라 한줄로 쭉세워라

        optimizer.zero_grad()#왜먼저했을까? 상관없기때문에 ㅋㅋ
        hypothesis = linear(X)#실제 Y랑 비교해서 확률 log함수를 구하는거
        cost = criterion(hypothesis, Y) 
        cost.backward()#그래디언트 방향
        optimizer.step()#그래디언트 이동

        avg_cost += cost / total_batch #코스트를 싹다더해서 평균적으로 얼마정도를 발생하는지를 구했는데 러닝할 땐 필요없고 내가 보고싶을 때씀

    print(f'Epoch: {epoch+1:04d}, cost={avg_cost:.9f}')

print('Learning finished')
# Test the model using test sets


# In[13]:


with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28 * 28).float()
    Y_test = mnist_test.test_labels

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test #얼마나 일치하니?
    accuracy = correct_prediction.float().mean()#트루의 비율이 얼마나 되니?
    print('Accuracy:', accuracy.item())


# In[ ]:




