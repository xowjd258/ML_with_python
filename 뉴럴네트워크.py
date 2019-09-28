#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch #토치 라이브러리
import torch.nn as nn #nn관련 라이브러리
import torch.nn.functional as F #활성화 함수, 데이터가 들어왔다고해서 무조건 출력하는것이아니라 출력에 대한 값을 조절(다양한 함수 포함)


class Net(nn.Module): #상속

    def __init__(self): #계층을 나누는 역할
        super(Net, self).__init__() #다량의 상속 발생 방지 , 일반적인 네트워크 쓸 때 복붙가능
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)# 1계층,(1채널->6채널, 5*5행렬, 이미지 처리할 때 3채널에서 1채널로가면 많은속설 상실)
        self.conv2 = nn.Conv2d(6, 16, 5)# 2계층,(6채널->16채널,5*5행렬)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  #3계층: 받은 입력을 1자로 flat 해서 받음, 120개를 출력하겠다
        self.fc2 = nn.Linear(120, 84) #4계층: 받은 120개의 입력을 84개의 입력으로 바꾸고
        self.fc3 = nn.Linear(84, 10) # 5계층: 받은 74개의 입력을 최종적으로 10개 출력

    def forward(self, x): #계층을 이어주는 역할
        #  (2,2)에서 maxpool을 적용을 하겠다,(2,2)사이즈로 나눠서 가장큰값을 가져오겠다!
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) #relu: 활성화 함수, max_pool2d 최대치의 값을 함축
       
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size()) #첫 번째 계층에서 처리했던 방식

list(net.parameters()) # 어떤 알고리즘으로 분석했는지를 상세하게 알려줌, 훈련시킨적이없기 때문에 랜덤하게 나옴


