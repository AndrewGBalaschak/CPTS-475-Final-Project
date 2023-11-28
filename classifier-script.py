import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"

class NFUQNIDS(Dataset):
    def __init__(self, data_file, transform=None, target_transform=None):
        self.data = pd.read_csv(data_file)
        self.labels = self.data['Label']
        self.data = self.data.drop({'Label', 'Attack', 'Dataset'}, axis=1)
        self.data = self.data.values

        # Standardize the features
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = self.labels.iloc[idx]
        if self.transform:
            item = self.transform(item)
        if self.target_transform:
            label = self.target_transform(label)
        return torch.tensor(item, dtype=torch.float32).to(device), torch.tensor(label, dtype=torch.float32).to(device)

dataset = NFUQNIDS('data/cleaned/NF-UQ-NIDS-CLEANED.csv')

# Train/Test split
train_size = int(0.8 * dataset.__len__())
test_size = dataset.__len__() - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 512
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Model Hyperparameters
input_dim = 75
hidden_dim = 256
output_dim = 1
learning_rate = 0.001

# Model Definition
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for binary classification
        return x

model = Classifier(input_dim, hidden_dim, output_dim).to(device)

# Optimizer Definition
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
num_epochs = 100

# Lists to store the loss values for training and validation sets
train_loss_list = []
test_loss_list = []

for epoch in tqdm(range(num_epochs)):
    epoch_train_loss = 0
    epoch_test_loss = 0

    # Train the model
    for X_batch, y_batch in train_data:
        # Calculate training output and loss
        train_outputs = model(X_batch).squeeze()
        train_loss = criterion(train_outputs, y_batch)
        epoch_train_loss = epoch_train_loss + train_loss.item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    # Calculate training loss this epoch
    train_loss_list.append(epoch_train_loss / len(train_data))

    # Print loss to command line
    if(((epoch+1) % 10) == 0):
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {(epoch_train_loss / len(train_data)):.4f}')

    # Test the model
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in train_data:
            # Calculate training output and loss
            train_outputs = model(X_batch).squeeze()
            train_loss = criterion(train_outputs, y_batch)
            
            epoch_test_loss = epoch_test_loss + train_loss.item()

    # Calculate testing loss this epoch
    test_loss_list.append(epoch_test_loss / len(test_data))

# Calculate statistics
# accuracy = accuracy_score(y_test.cpu(), predictions.cpu())
# precision = precision_score(y_test.cpu(), predictions.cpu())
# recall = recall_score(y_test.cpu(), predictions.cpu())
# confusion = confusion_matrix(y_test.cpu(), predictions.cpu())

# print(f'Accuracy: {accuracy:.2f}')
# print(f'Precision: {precision:.2f}')
# print(f'Recall: {recall:.2f}')
# print(f'Confusion Matrix:\n{confusion}')

# Plot the loss curve
plt.plot(train_loss_list, label="train")
plt.plot(test_loss_list, label="validation")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()