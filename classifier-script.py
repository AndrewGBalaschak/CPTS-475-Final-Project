import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# device = "cuda" if torch.cuda.is_available() else "cpu"
# This model is too big for my gpu
device = 'cpu'

data = pd.read_csv('data/cleaned/NF-UQ-NIDS-CLEANED.csv')
data = data.head(500000)

# Separate features (X) and labels (y)
X = data.drop({'Label', 'Attack', 'Dataset'}, axis=1)
y = data['Label']

# Delete data so it can be garbage collected
del data

# Drop column labels
X = X.values
y = y.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

# Delete data so it can be garbage collected
del X
del y

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Model Hyperparameters
input_dim = X_train.shape[1]
hidden_dim = 256
output_dim = 1

# Model Definition
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Sigmoid activation for binary classification
        return x

model = Classifier(input_dim, hidden_dim, output_dim).to(device)

# Optimizer Definition
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 100

# Lists to store the loss values for training and validation sets
train_loss_list = []
test_loss_list = []

for epoch in tqdm(range(num_epochs)):
    train_outputs = model(X_train).squeeze()
    train_loss = criterion(train_outputs, y_train)
    train_loss_list.append(train_loss.item())

    test_outputs = model(X_test).squeeze()
    test_loss = criterion(test_outputs, y_test)
    test_loss_list.append(test_loss.item())

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    if(((epoch+1) % 10) == 0):
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

del train_outputs
del test_outputs
del train_loss
del test_loss

# Plot the loss curve
plt.plot(train_loss_list, label="train")
plt.plot(test_loss_list, label="validation")
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predictions = test_outputs.max(1)
    
    accuracy = accuracy_score(y_test.cpu(), predictions.cpu())
    confusion = confusion_matrix(y_test.cpu(), predictions.cpu())

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Confusion Matrix:\n{confusion}')