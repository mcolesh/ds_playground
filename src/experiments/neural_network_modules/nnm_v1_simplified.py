import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer


class BinaryClassification(torch.nn.Module):
    def __init__(self, input_dimension):
        super().__init__()
        self.linear = torch.nn.Linear(input_dimension, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


data = load_breast_cancer()
X, Y = data.data, data.target

"""let's preprocess, normalize and create the model"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_test.shape)
num_of_samples, input_dimension = X_train.shape
model = BinaryClassification(input_dimension)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1))
y_test = torch.from_numpy(y_test.astype(np.float32).reshape(-1, 1))


criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters())
num_of_samples, input_dimension = X_train.shape
model = BinaryClassification(input_dimension)
train_losses, test_losses = train(model, criterion, optimizer, X_train, y_train)


plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.show()

train_acc, test_acc = evaluate(model, X_train, y_train, X_test, y_test)
print(train_acc)
