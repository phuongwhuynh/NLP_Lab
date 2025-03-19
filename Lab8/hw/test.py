import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
with zipfile.ZipFile("digit-recognizer.zip", "r") as zip_ref:
    zip_ref.extractall(".")  
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
X = train.drop("label", axis=1).values / 255.0  # Normalize
y = train["label"].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)




transform = transforms.Compose([
    transforms.RandomRotation(15),        # Rotate ±15 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Shift by 10%
    transforms.RandomHorizontalFlip(),    # Horizontal Flip (not useful for digits, but added for variety)
    transforms.RandomErasing(p=0.3)       # Randomly erase part of the image
])

def apply_transform(X):
    X = X.view(-1, 1, 28, 28)  # Reshape to (N, C, H, W)
    X = transform(X)           # Apply transforms
    X = X.view(-1, 784)        # Flatten back
    return X


def generate_augmented_data(X, y, num_augments=2):
    X_aug, y_aug = [], []

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    for _ in range(num_augments):
        X_transformed = apply_transform(X_tensor)
        X_aug.append(X_transformed)
        y_aug.append(y_tensor)

    X_aug = torch.cat([X_tensor] + X_aug)  # Stack original + augmented
    y_aug = torch.cat([y_tensor] + y_aug)

    return X_aug.numpy(), y_aug.numpy()

X_train_aug, y_train_aug = generate_augmented_data(X_train, y_train, num_augments=2)  # Doubles dataset




X_train_tensor = torch.tensor(X_train_aug, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_aug, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(test.values / 255.0, dtype=torch.float32)

batch_size=128
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor),batch_size=batch_size, shuffle=False)


def swish(x):
    return x * torch.sigmoid(x)


# class MLP(nn.Module):
#     def __init__(self, input_size=784,hidden_size=512,output_size=10,dropout_rate=0.3):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.bn1 = nn.BatchNorm1d(hidden_size)
#         self.dropout1 = nn.Dropout(dropout_rate)

#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.bn2 = nn.BatchNorm1d(hidden_size)
#         self.dropout2 = nn.Dropout(dropout_rate)

#         self.fc3 = nn.Linear(hidden_size, hidden_size)
#         self.bn3 = nn.BatchNorm1d(hidden_size)
#         self.dropout3 = nn.Dropout(dropout_rate)

#         self.fc4 = nn.Linear(hidden_size, hidden_size)
#         self.bn4 = nn.BatchNorm1d(hidden_size)
#         self.dropout4=nn.Dropout(dropout_rate)

#         self.fc5 = nn.Linear(hidden_size, hidden_size)
#         self.bn5 = nn.BatchNorm1d(hidden_size)
#         self.dropout5=nn.Dropout(dropout_rate)


#         self.fc_out = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x1 = torch.relu(self.bn1(self.fc1(x)))
#         x1 = self.dropout1(x1)
        
#         x2 = torch.relu(self.bn2(self.fc2(x1)))
#         x2 = self.dropout2(x2)
#         x2 = x1 + x2  # Skip connection
        
#         x3 = torch.relu(self.bn3(self.fc3(x2)))
#         x3 = self.dropout3(x3)
#         x3 = x2 + x3  # Skip connection

#         x4 = torch.relu(self.bn4(self.fc4(x3)))
#         x4 = self.dropout4(x4)
#         x4 = x3 + x4  # Skip connection

#         x5 = torch.relu(self.bn5(self.fc5(x4)))
#         x5 = self.dropout4(x5)
#         x5 = x4 + x5  # Skip connection

#         # Output layer (Softmax applied later in loss function)
#         out = self.fc_out(x5)
#         return out
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=1024, output_size=10, dropout_rate=0.3):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for _ in range(9):  # 9 additional hidden layers
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_res = swish(self.bn1(self.fc1(x)))
        x_res = self.dropout1(x_res)
        
        for i in range(9):
            x_new = swish(self.batch_norms[i](self.hidden_layers[i](x_res)))
            x_new = self.dropouts[i](x_new)
            x_res = x_res + x_new  # Skip connection
        
        out = self.fc_out(x_res)
        return out


model = MLP(hidden_size=512).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  

epochs=200
for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (outputs.argmax(1) == y_batch).sum().item()

    train_acc = correct / len(X_train_aug)
    
    # Validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            outputs = model(X_val_batch)
            correct += (outputs.argmax(1) == y_val_batch).sum().item()
    
    val_acc = correct / len(X_val)
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

    # Adjust learning rate
    scheduler.step()

model.eval()
predictions = []
with torch.no_grad():
    for X_test_batch in test_loader:
        X_test_batch = X_test_batch[0].to(device)
        outputs = model(X_test_batch)
        predictions.extend(outputs.argmax(1).cpu().numpy())

# Create submission file
submission = pd.DataFrame({"ImageId": np.arange(1, len(predictions) + 1), "Label": predictions})
submission.to_csv("submission.csv", index=False)
