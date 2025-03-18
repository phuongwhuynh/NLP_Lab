import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Preprocess data
X = train_data.iloc[:, 1:].values / 255.0  # Normalize to [0, 1]
y = train_data.iloc[:, 0].values
X_test = test_data.values / 255.0

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Define the model
class MLPWithSkip(nn.Module):
    def __init__(self):
        super(MLPWithSkip, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

    def forward(self, x):
        # Layer 1
        out = self.fc1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.dropout(out)

        # Layer 2 with skip connection
        identity = out
        out = self.fc2(out)
        out = self.bn2(out)
        out = nn.ReLU()(out + identity)  # Skip connection
        out = self.dropout(out)

        # Layer 3
        out = self.fc3(out)
        out = self.bn3(out)
        out = nn.ReLU()(out)
        out = self.dropout(out)

        # Layer 4 with skip connection
        identity = out
        out = self.fc4(out)
        out = self.bn4(out)
        out = nn.ReLU()(out + identity)  # Skip connection
        out = self.dropout(out)

        # Layer 5
        out = self.fc5(out)
        out = self.bn5(out)
        out = nn.ReLU()(out)
        out = self.dropout(out)

        # Output layer
        out = self.fc6(out)
        return out

# Initialize model, loss, and optimizer
model = MLPWithSkip()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100*correct/total:.2f}%")

# Test-time augmentation (optional)
def test_time_augmentation(model, x, n_augments=10):
    model.eval()
    outputs = []
    for _ in range(n_augments):
        # Apply random transformations (e.g., small rotations, shifts)
        angle = np.random.uniform(-10, 10)  # Random rotation between -10 and 10 degrees
        shift = np.random.uniform(-2, 2, size=2)  # Random shift between -2 and 2 pixels
        # Normalize shift to be within [0, 1] for translate parameter
        translate = (shift[0] / 28, shift[1] / 28)  # Divide by image size (28x28)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=(angle, angle), translate=translate),
            transforms.ToTensor(),
        ])
        x_aug = transform(x)
        x_aug = x_aug.view(1, -1).to(device)
        output = model(x_aug)
        outputs.append(output)
    return torch.mean(torch.stack(outputs), dim=0)

# Evaluate on test data
model.eval()
test_predictions = []
with torch.no_grad():
    for i in range(len(X_test)):
        x = torch.tensor(X_test[i], dtype=torch.float32).view(1, -1).to(device)
        output = test_time_augmentation(model, x)  # Use TTA for better accuracy
        _, predicted = torch.max(output.data, 1)
        test_predictions.append(predicted.item())

# Save predictions
submission = pd.DataFrame({"ImageId": range(1, len(test_predictions)+1), "Label": test_predictions})
submission.to_csv("submission.csv", index=False)