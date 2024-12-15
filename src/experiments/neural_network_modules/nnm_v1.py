"""implementation of a basic neural network for binary dicision"""

import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim


class BinaryClassification(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassification, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # Fully connected layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# Training function
def train_model(model, criterion, optimizer, train_loader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            # Move data to GPU if available
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")


# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {acc:.4f}")


# Example usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    # Sample data
    X_train = torch.rand((100, 10))  # 100 samples, 10 features
    y_train = torch.randint(0, 2, (100,))  # Binary labels

    X_test = torch.rand((20, 10))
    y_test = torch.randint(0, 2, (20,))

    # Data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model, loss, and optimizer
    input_dim = X_train.shape[1]
    model = BinaryClassification(input_dim)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    train_model(model, criterion, optimizer, train_loader, num_epochs=10)
    evaluate_model(model, test_loader)
