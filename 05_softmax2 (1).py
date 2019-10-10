import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
    
# initialize
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3) # output dimension = 3

    def forward(self, x):
        return self.linear(x)


model = SoftmaxClassifierModel()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

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
