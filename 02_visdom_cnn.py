#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as dsets


# In[18]:


import visdom
vis = visdom.Visdom()
vis.close(env="main")


# In[19]:


def loss_tracker(loss_plot, loss_value, num):
    '''num, loss_value, are Tensor'''
    vis.line(X=num,
             Y=loss_value,
             win = loss_plot,
             update='append'
             )


# In[20]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device =='cuda':
    torch.cuda.manual_seed_all(777)
print(f"using {device}")


# In[21]:


#parameters
learning_rate = 0.001
training_epochs = 15 
batch_size = 32 #당연히 한번에 여러개를 다루는게 빠릴 수렴


# In[22]:


# Download CIFAR10 dataset
# 60000 images 10 classes (6000 images per class)
cifar10_train = dsets.CIFAR10(root="./cifar10",
                              train = True,#트레인이 트루냐
                              transform=torchvision.transforms.ToTensor(),
                              download=True)
cifar10_test = dsets.CIFAR10(root="./cifar10",
                              train = False,#트레인이 False면 테스트데이터
                              transform=torchvision.transforms.ToTensor(),
                              download=True)


# In[23]:


data_loader = torch.utils.data.DataLoader(dataset=cifar10_train,
                                          batch_size = batch_size,
                                          shuffle =True,
                                          drop_last=True) #트레인을 도와줄 데이터 로더
                                            #드랍라스트는 남는건 버린다


# In[24]:


class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3, stride=1, padding=1),#32*32 RGB 세 장, 한칸씩이동, 한줄 0으로 두른다
            nn.ReLU(),#2*2마다 숫자 한 개씩 뽑는다
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(4*4*128, 625) #풀커넥트 리니어레이어
        self.relu = nn.ReLU()#시그모이드
        self.fc2 = nn.Linear(625, 10, bias =True)#625개의 레이어를 10개로 줄여 줌
        torch.nn.init.xavier_uniform_(self.fc1.weight)#웨이트 노말라이즈
        torch.nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x):     #구조 레이어1->레이어2->레이어3
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# In[25]:


model = CNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# In[26]:


loss_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
test_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='test_tracker', legend=['test'], showlegend=True))


# In[ ]:


#training
total_batch = len(data_loader)

test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size,#아까는 트레이닝로더 지금은 테스터 로드
                                          shuffle=False, drop_last=True)
total_test_batch = len(test_loader)


for epoch in range(training_epochs):
    model.train()#트레인할때는 나의 모드는 트레이닝이다
    avg_cost = 0
    
    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)
        
        optimizer.zero_grad()
        hypothesis = model(X)
        
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        
        avg_cost += cost / total_batch
    
    print('[Epoch:{}] cost = {}'.format(epoch+1, avg_cost))#트래커를 이용해 로스트래킹
    loss_tracker(loss_plt, torch.Tensor([avg_cost]), torch.Tensor([epoch]))
    
    model.eval()#테스트할때는 나의모델은 이벨류에이션
    avg_acc = 0
    with torch.no_grad():#내가지금 테스트할꺼니까 그라드 계산하지마라
        for X, Y in test_loader:
            X_test = X.to(device)
            Y_test = Y.to(device)
    
            prediction = model(X_test)
            correct_prediction = torch.argmax(prediction, 1) == Y_test
            accuracy = correct_prediction.float().mean() 
            avg_acc += accuracy/total_test_batch
        print('Accuracy:', avg_acc.item())
        loss_tracker(test_plt, torch.Tensor([avg_acc]), torch.Tensor([epoch]))

print('Learning Finished!')


# In[ ]:




