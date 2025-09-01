import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# ---------- Paths ----------
FEATURES_DIR = "features"
MODEL_PATH = "somali_alphabet_asr.pth"

# ---------- Somali Alphabet ----------
labels = [
    'b','t','j','x','kh','d','r','s','sh','dh',
    'c','g','f','q','k','l','m','n','w','h','y',
    'a','e','i','o','u'
]
label2idx = {lab: i for i, lab in enumerate(labels)}

# ---------- Load features ----------
X, y = [], []

for file in os.listdir(FEATURES_DIR):
    if file.endswith(".npy"):
        mfcc = np.load(os.path.join(FEATURES_DIR, file))

        # Average across time frames → fixed-size vector
        mfcc_mean = np.mean(mfcc, axis=1)
        X.append(mfcc_mean)

        # Extract label from filename
        label = file.split("_")[0]
        y.append(label2idx[label])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print(f"Total samples: {len(y)}")
print(f"Feature shape: {X.shape}")

# ---------- Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# ---------- Neural Network ----------
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

input_dim = X_train.shape[1]
hidden_dim = 64
num_classes = len(labels)
model = SimpleNN(input_dim, hidden_dim, num_classes)

# ---------- Loss and Optimizer ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------- Training Loop ----------
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1)
        acc = (preds == y_test).float().mean().item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Acc: {acc:.2f}")

# ---------- Save Model ----------
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
