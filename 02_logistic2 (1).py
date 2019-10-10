import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]] #데이터 선언
y_data = [[0], [0], [0], [1], [1], [1]] 
x_train = torch.FloatTensor(x_data) #리스트데이터를 파이토치가 이해할 수 있게 플로트텐서 데이터로 바꿔줌
y_train = torch.FloatTensor(y_data) #위와 동일

class BinaryClassifier(nn.Module): #클래스 선언(뉴럴넷. 모듈)
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): #내 x를
        return self.sigmoid(self.linear(x)) #리니어에 한번 통과시키고 시그모이드에 한번통과시킨다(리니어는 아까 행렬곱)
    #x1,x2->Linear->a1x1+a2x2+b->sigmoid->1/(1+e^(-a1x1-a2x2-b))

model = BinaryClassifier() #이 과정이 내 모델이다


# optimizer
optimizer = optim.SGD(model.parameters(), lr=1) #모델안에 있는 파라미터들을 옵티마이즈 하겠다

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # cost
    hypothesis = model(x_train) #나의 예측값은 모델이다
    cost = F.binary_cross_entropy(hypothesis, y_train) #Loss함수를 계산하는 바이너르크로스엔트로피

    # gradient descent
    optimizer.zero_grad() #초기화
    cost.backward() #미분값계산
    optimizer.step() #그방향으로 이동

    # check progress
    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')
