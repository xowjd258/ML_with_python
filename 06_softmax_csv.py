import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)

x_train = torch.FloatTensor(xy[:, 0:-1])
# squeeze : 2d->1d
y_train = torch.LongTensor(xy[:, [-1]]).squeeze()

# initialize
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 7) # output dimension = 3

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
