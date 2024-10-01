
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
 
n_feats = 28*28
n_hidden = 256
n_code = 64

# Define the autoencoder
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_code),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_code, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_feats),
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
            BitLinear(n_feats, n_hidden, **self.bitlinear_kwargs),
            nn.ReLU(),
            BitLinear(n_hidden, n_hidden, **self.bitlinear_kwargs),
            nn.ReLU(),
            BitLinear(n_hidden, n_code, **self.bitlinear_kwargs),
        )
        self.decoder = nn.Sequential(
            BitLinear(n_code, n_hidden, **self.bitlinear_kwargs),
            nn.ReLU(),
            BitLinear(n_hidden, n_hidden, **self.bitlinear_kwargs),
            nn.ReLU(),
            BitLinear(n_hidden, n_feats, **self.bitlinear_kwargs)
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class HybridAE(nn.Module):
    def __init__(self, weight_measure="AbsMean", bias=True):
        super(HybridAE, self).__init__()

        self.bitlinear_kwargs = { "weight_measure": weight_measure, "bias": bias }
        self.encoder = nn.Sequential(
            BitLinear(n_feats, n_hidden, **self.bitlinear_kwargs),
            nn.SiLU(),
            BitLinear(n_hidden, n_code, **self.bitlinear_kwargs),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_code, n_feats),
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




# ae = AE()
# train(ae, train_loader, test_loader, n_epochs=20, lr=1e-3)


# bitae = BitAE()
# train(bitae, train_loader, test_loader, n_epochs=20, lr=1e-3)

hybridae = HybridAE()
print(hybridae)
train(hybridae, train_loader, test_loader, n_epochs=20, lr=1e-3)




# Notes
# =====
# First setting with incremental reduction up to 3 dimensions, interweaved with ReLU
# AE w/ MSE obj gets to ~~ 0.03 val/loss after 20 epochs
# BitAE w/ MSE obj gets to ~~ 0.06 val/loss after 20 epochs
# AE 2x better than BitAE

#####################################################
# Code dim: 64, corresponds to ~12x compression ratio

# **Linear** reduction to 64 dims
# AE w/ MSE obj gets to ~~ 0.009 val/loss after 20 epochs
# BitAE w/ MSE obj gets to ~~ 0.09 val/loss after 20 epochs
# AE 10x better than BitAE

# **Nonlinear** reduction to 64 dims, with one hidden layer of 128 units (ReLUs)
# AE w/ MSE obj gets to ~~ 0.0085 val/loss after 20 epochs
# BitAE w/ MSE obj gets to ~~ 0.055 val/loss after 20 epochs
# AE 6x better than BitAE

# **Nonlinear** reduction to 64 dims, with two hidden layers (128, ReLU) in-between
# AE w/ MSE obj gets to ~~  0.0086 val/loss after 20 epochs
# BitAE w/ MSE obj gets to ~~ 0.053 val/loss after 20 epochs
# AE 6x better than BitAE

# Same setting but w/ Linear decoder (HybridAE)
# HybridAE w/ 768-128-64 linear, then linear decoder ~~ 0.009 after 20 epochs 

# HybridAE w/ 768-128-64 bitlinear, then linear decoder ~~ 0.013 after 20 epochs 
# HybridAE w/ 768-128-64 bitlinear w/ SiLU act, then linear decoder ~~ 0.013 after 20 epochs 

# HybridAE w/ 768-64 linear, then linear decoder ~~ 0.009 after 20 epochs 
# HybridAE w/ 768-64 bitlinear, then linear decoder ~~ 0.014 after 20 epochs 
# HybridAE w/ 768-512-256-128-64 bitlinear w/ SiLU, then linear decoder ~~ 0.012 after 20 epochs 
# HybridAE w/ 768-256-64 bitlinear w/ SiLU, then linear decoder ~~ 0.012 after 20 epochs 

#####################################################