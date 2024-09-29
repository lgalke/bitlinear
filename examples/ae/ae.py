
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt

from bitlinear import BitLinear
 
# Download the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
 


# Define the autoencoder
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28)
            # nn.Sigmoid()
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Define the BitNet Autoencoder
class BitAE(nn.Module):
    def __init__(self, weight_measure="AbsMean", bias=True):
        super(BitAE, self).__init__()

        self.bitlinear_kwargs = { "weight_measure": weight_measure, "bias": bias }
        self.encoder = nn.Sequential(
            BitLinear(28*28, 128, **self.bitlinear_kwargs),
            nn.ReLU(),
            BitLinear(128, 64, **self.bitlinear_kwargs),
            nn.ReLU(),
            BitLinear(64, 12, **self.bitlinear_kwargs),
            nn.ReLU(),
            BitLinear(12, 3, **self.bitlinear_kwargs)
        )
        self.decoder = nn.Sequential(
            BitLinear(3, 12, **self.bitlinear_kwargs),
            nn.ReLU(),
            BitLinear(12, 64, **self.bitlinear_kwargs),
            nn.ReLU(),
            BitLinear(64, 128, **self.bitlinear_kwargs),
            nn.ReLU(),
            BitLinear(128, 28*28, **self.bitlinear_kwargs)
            # nn.Sigmoid()
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Define the training loop
def train(model, train_loader, test_loader, n_epochs=10, lr=1e-3):
    # criterion = nn.MSELoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    test_loss = []
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            img, _ = data
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss.append(running_loss / len(train_loader))
        print(f"Epoch {epoch+1}, Train Loss: {running_loss / len(train_loader)}")
        model.eval()
        running_loss = 0.0
        for data in test_loader:
            img, _ = data
            img = img.view(img.size(0), -1)
            outputs = model(img)
            loss = criterion(outputs, img)
            running_loss += loss.item()
        test_loss.append(running_loss / len(test_loader))
        print(f"Epoch {epoch+1}, Test Loss: {running_loss / len(test_loader)}")
    return train_loss, test_loss

model = AE()
train(model, train_loader, test_loader, n_epochs=20, lr=1e-3)