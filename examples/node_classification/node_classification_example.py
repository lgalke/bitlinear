import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SGConv, GCNConv

import torch.nn.functional as F

from bitlinear import BitLinear, replace_modules


DATASET_KEY = "PubMed"  # Cora, Citeseer, PubMed
MODEL_KEY = "BitGCNConv"  # SGConv, BitSGConv, GCNConv, BitGCNConv
BITLINEAR_WEIGHT_MEASURE = "AbsMedian"  # AbsMean, AbsMedian

GCN_N_HIDDEN = 32
GCN_IMPROVED = False

K = 2  # Only relevant for SGConv and BitSGConv
lr = 0.01


dataset = Planetoid(root=f"/tmp/Planetoid/{DATASET_KEY}", name=DATASET_KEY)


class BitSGConv(SGConv):
    def __init__(self, in_channels: int, out_channels: int, K=1, bias=True, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(in_channels, out_channels, K=K, bias=bias, **kwargs)
        self.lin = BitLinear(
            in_channels,
            out_channels,
            bias=bias,
            weight_measure=BITLINEAR_WEIGHT_MEASURE,
        )
        self.reset_parameters()


class BitGCNConv(GCNConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved=False,
        add_self_loops=None,
        normalize=True,
        bias=True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(
            in_channels,
            out_channels,
            improved=improved,
            add_self_loops=add_self_loops,
            normalize=True,
            bias=bias,
            **kwargs,
        )
        self.lin = BitLinear(
            in_channels,
            out_channels,
            bias=bias,
            weight_measure=BITLINEAR_WEIGHT_MEASURE,
        )
        self.reset_parameters()



class BitGCN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BitGCN, self).__init__()
        self.conv1 = BitGCNConv(in_channels, GCN_N_HIDDEN, **kwargs)
        self.conv2 = BitGCNConv(GCN_N_HIDDEN, out_channels, **kwargs)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        y = self.conv2(h, edge_index)
        return y

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, GCN_N_HIDDEN, **kwargs)
        self.conv2 = GCNConv(GCN_N_HIDDEN, out_channels, **kwargs)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        y = self.conv2(h, edge_index)
        return y

def build_model():
    # Initialize the model and optimizer
    sgc_kwargs = {"K": K, "add_self_loops": True, "cached": True, "aggr": "sum"}
    gcn_kwargs = {
        "improved": GCN_IMPROVED,
        "add_self_loops": True,
        "cached": True,
        "aggr": "sum",
    }

    if MODEL_KEY == "SGConv":
        model = SGConv(dataset.num_features, dataset.num_classes, **sgc_kwargs)
    elif MODEL_KEY == "BitSGConv":
        model = BitSGConv(dataset.num_features, dataset.num_classes, **sgc_kwargs)
    elif MODEL_KEY == "GCNConv":
        model = GCN(dataset.num_features, dataset.num_classes, **gcn_kwargs) 
    elif MODEL_KEY == "BitGCNConv":
        model = BitGCN(dataset.num_features, dataset.num_classes, **gcn_kwargs)
    else:
        raise ValueError(f"Unknown MODEL_KEY: {MODEL_KEY}")

    return model


# Training loop
def train(model):
    model.train()
    optimizer.zero_grad()
    out = model(dataset[0].x, dataset[0].edge_index)
    loss = criterion(out[dataset[0].train_mask], dataset[0].y[dataset[0].train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


# Evaluation loop
def test(model):
    model.eval()
    out = model(dataset[0].x, dataset[0].edge_index)
    pred = out.argmax(dim=1)
    acc = pred[dataset[0].test_mask] == dataset[0].y[dataset[0].test_mask]
    acc = int(acc.sum()) / int(dataset[0].test_mask.sum())
    return acc


# Train and evaluate the model

final_accuracy_scores = []
epochs = 100
for run in range(10, 11):
    model = build_model()
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train_loss = train(model)
        acc = test(model)
        print(
            f"Epoch: {epoch+1}, Train loss: {train_loss:.4f}, Test Accuracy: {acc:.4f}"
        )
    final_accuracy_scores.append(acc)

    
    MODEL_NAME = f"{MODEL_KEY}-{BITLINEAR_WEIGHT_MEASURE}" if MODEL_KEY.startswith("Bit") else MODEL_KEY
    with open("results.csv", "a") as f:
        f.write(f"{run},{MODEL_NAME},,{lr},{epochs},{DATASET_KEY},{acc}\n")

print("**********************")
print("Final accuracy scores:")
print(DATASET_KEY, MODEL_NAME)
print(final_accuracy_scores)
print("**********************")
