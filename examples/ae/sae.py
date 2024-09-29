import math
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.linalg import vector_norm
from torch.utils.data import DataLoader

from typing import Union

import scipy.spatial.distance as dist
import scipy.stats as stats 

from bitlinear import BitLinear

""" 

BitNet Sparse Autoencoder implementation

&

For comparison, a Sparse Autoencoder implementation
Inspired by: 
[1] https://transformer-circuits.pub/2023/monosemantic-features/index.html
[2] ICA with Reconstruction Cost for Efficient Overcomplete Feature Learning
"""

class BitSAE(nn.Module):
    def __init__(
        self,
        n_feature: int,
        n_hidden: int,
        use_pre_encoder_bias: bool = True,
        initial_decoder_bias: torch.Tensor = None,
        bitlinear_kwargs: dict = {},
    ):
        super(BitSAE, self).__init__()
        self.encoder = BitLinear(n_feature, n_hidden, **bitlinear_kwargs)
        self.decoder = BitLinear(n_hidden, n_feature, **bitlinear_kwargs)
        self.use_pre_encoder_bias = use_pre_encoder_bias
        if initial_decoder_bias is not None:
            self.set_decoder_bias(initial_decoder_bias)

        # Not needed when using BitLinear
        # self.normalize_decoder_weight()

    def forward(self, x: torch.Tensor):
        # pre-encoder bias, as in [1]
        if self.use_pre_encoder_bias:
            x = x - self.decoder.bias
        f = F.relu(self.encoder(x))
        x_hat = self.decoder(f)
        return x_hat, f

    @torch.no_grad()
    def set_decoder_bias(self, bias: torch.Tensor):
        self.decoder.bias.copy_(bias)

    @torch.no_grad()
    def normalize_decoder_weight(self):
        self.decoder.weight.data[:] = self.decoder.weight.data / vector_norm(
            self.decoder.weight.data, ord=2, dim=0, keepdim=True
        )

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        # adapted from here: https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/bcae01328a2f41d24bd4a9160828f2fc22737f75/utils.py#L135
        # but dim=0 since our weight is transposed
        # and we don't need to assign the normalized the weight
        # because we do it in normalize_decoder_weight
        W_dec = self.decoder.weight
        W_dec_normed = W_dec / vector_norm(W_dec, ord=2, dim=0, keepdim=True)
        W_dec_grad_proj = (W_dec.grad * W_dec_normed).sum(0, keepdim=True) * W_dec_normed
        self.decoder.weight.grad -= W_dec_grad_proj




class SAE(nn.Module):
    def __init__(
        self,
        n_feature: int,
        n_hidden: int,
        use_pre_encoder_bias: bool = True,
        initial_decoder_bias: torch.Tensor = None,
    ):
        super(SAE, self).__init__()
        self.encoder = nn.Linear(n_feature, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_feature)
        self.use_pre_encoder_bias = use_pre_encoder_bias
        if initial_decoder_bias is not None:
            self.set_decoder_bias(initial_decoder_bias)

        self.normalize_decoder_weight()

    def forward(self, x: torch.Tensor):
        # pre-encoder bias, as in [1]
        if self.use_pre_encoder_bias:
            x = x - self.decoder.bias
        f = F.relu(self.encoder(x))
        x_hat = self.decoder(f)
        return x_hat, f

    @torch.no_grad()
    def set_decoder_bias(self, bias: torch.Tensor):
            self.decoder.bias.copy_(bias)

    @torch.no_grad()
    def normalize_decoder_weight(self):
        self.decoder.weight.data[:] = self.decoder.weight.data / vector_norm(
            self.decoder.weight.data, ord=2, dim=0, keepdim=True
        )

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        # adapted from here: https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/bcae01328a2f41d24bd4a9160828f2fc22737f75/utils.py#L135
        # but dim=0 since our weight is transposed
        # and we don't need to assign the normalized the weight
        # because we do it in normalize_decoder_weight
        W_dec = self.decoder.weight
        W_dec_normed = W_dec / vector_norm(W_dec, ord=2, dim=0, keepdim=True)
        W_dec_grad_proj = (W_dec.grad * W_dec_normed).sum(0, keepdim=True) * W_dec_normed
        self.decoder.weight.grad -= W_dec_grad_proj


class SAELoss(nn.Module):
    """Sparse Autoencoder loss function

    Args:
        l1_coefficient (float): L1 regularization coefficient
        rescale (bool): whether to rescale the L1 loss by code dimension

    rescale is added to make the L1 loss comparable across different code dimensions,
    such that l1_coefficient does not need to be adjusted when changing the code dimension.
    """

    def __init__(self, l1_coefficient: float = 1.0, rescale: bool = True):
        super(SAELoss, self).__init__()
        self.l1_coefficient = float(l1_coefficient)
        self.rescale = bool(rescale)

    def forward(self, x_hat: torch.Tensor, f: torch.Tensor, x: torch.Tensor):
        rec_loss = F.mse_loss(x_hat, x, reduction="mean")
        # v1 normalize by number of activations
        # sps_loss = torch.linalg.norm(f, ord=1, dim=-1, keepdim=True).sum() / f.numel()

        # v2 (as in [1]) mean over dataset
        sps_loss = vector_norm(f, ord=1, dim=-1, keepdim=True).mean()
        # with rescale, it is equivalent to v1 above
        if self.rescale:
            sps_loss = sps_loss / f.size(-1)

        loss = rec_loss + self.l1_coefficient * sps_loss
        return loss, rec_loss, sps_loss


def train_sae(
    model: SAE,
    data: Union[DataLoader, torch.Tensor],
    lr: float = 3e-4,
    beta1=0.9,  # default Adam parameters
    beta2=0.999,
    l1_coefficient: float = 1.0,
    batch_size: int = 1024,
    epochs: int = 20000,
    meaning_space: torch.Tensor = None,
    force_unit_decoder_norm=True,
) -> None:
    device = next(model.parameters()).device
    criterion = SAELoss(l1_coefficient, rescale=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

    if isinstance(data, DataLoader):
        loader = data
    else:
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    decay_start_epoch = int(epochs * 0.8)
    def decay_lr(e):
        return 1.0 if e < decay_start_epoch else (epochs - e) / (epochs - decay_start_epoch)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, decay_lr)

    for epoch in range(epochs):
        model.train()
        running_rec_loss = 0.0
        running_sps_loss = 0.0
        running_loss = 0.0

        for batch in loader:
            x = batch.to(device)
            optimizer.zero_grad()
            x_hat, f = model(x)
            loss, rec_loss, sps_loss = criterion(x_hat, f, x)
            loss.backward()

            if force_unit_decoder_norm:
                model.remove_parallel_component_of_grads()

            optimizer.step()

            if force_unit_decoder_norm:
                model.normalize_decoder_weight()

            running_rec_loss += rec_loss.item()
            running_sps_loss += sps_loss.item()
            running_loss += loss.item()

        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch}, loss: {running_loss / len(loader):.4f} | sps_loss: {running_sps_loss / len(loader):.4f} | rec_loss: {running_rec_loss / len(loader):.4f}"
            )
            f = sae_inference(model, data)
            avg_l0_norm, sp_frac = eval_l0_sparsity(f)
            print(f"\t-> Sparsity (fraction of zero activations): {sp_frac*100:.2f}%")
            print(f"\t-> Avg L0 norm of the code (avg. number active components per example): {avg_l0_norm:.4f}")
            decoder_weight_norm = vector_norm(model.decoder.weight, ord=2, dim=0)
            print(f"\t-> Mean/SD L2 decoder weight = {decoder_weight_norm.mean()} / {decoder_weight_norm.std()}")
            print(f"\tlr={scheduler.get_last_lr()}")
            if meaning_space is not None:
                topographic_similarity(f, meaning_space)

        scheduler.step()


def sae_inference(model: SAE, data: torch.Tensor) -> torch.Tensor:
    model.eval()
    data_device = data.device
    model_device = next(model.parameters()).device
    data = data.to(model_device)
    with torch.no_grad():
        __x_hat, f = model(data)
    return f.to(data_device)


# def eval_sparsity(z):
#     nz = torch.count_nonzero(z)
#     nz_frac = nz / z.numel()
#     print(nz_frac * z.numel() / z.size(0))
#     # avg_l0_norm = nz_frac * z.numel() / num_examples
#     sp_frac = 1.0 - nz_frac
#     return sp_frac


def eval_l0_sparsity(z: torch.Tensor) -> tuple[float, float]:
    l0_norm = vector_norm(z, ord=0, dim=-1)

    # Average by number of examples
    avg_l0_norm = l0_norm.mean()

    # Average by total number of elements (n examples * n features)
    nz_frac = l0_norm.sum() / z.numel()
    sp_frac = 1.0 - nz_frac

    return avg_l0_norm.item(), sp_frac.item()


def geometric_median(data: torch.Tensor) -> torch.Tensor:
    from geom_median.torch import compute_geometric_median

    gm_result = compute_geometric_median(data)
    ok = gm_result.termination == "function value converged within tolerance"
    if not ok:
        print(
            "Warning: geometric median computation did not converge: ",
            gm_result.termination,
        )
    gm = gm_result.median
    return gm



def topographic_similarity(X, Y):
    dist.pdist(X, 'euclidean')
    # v1
    X_pdist = dist.pdist(X, 'minkowski', p=1).ravel()
    Y_pdist = dist.pdist(Y, 'hamming').ravel()
    pearson = stats.pearsonr(X_pdist, Y_pdist)
    print("Topographic similarity:")
    print(pearson)
    spearman = stats.spearmanr(X_pdist, Y_pdist)
    print(spearman)




def main():
    num_examples = 4048 
    num_features = 128
    meaning_space = torch.randint(0, 2, (num_examples, num_features), dtype=torch.bool)
    print(meaning_space)
    print("L1 norm meaning space", vector_norm(meaning_space.float(), ord=0, dim=-1).mean()) # E[||x||_0] = num_features / 2

    half = num_features // 2

    polysemantic_dataset_and = (meaning_space[:, :half] & meaning_space[:, half:]).float()
    polysemantic_dataset_or = (meaning_space[:, :half] | meaning_space[:, half:]).float()

    polysemantic_dataset = torch.cat([polysemantic_dataset_and, polysemantic_dataset_or], dim=1).float()

    dataset = polysemantic_dataset

    print("L1 norm dataset", vector_norm(dataset, ord=0, dim=-1).mean())
    # input()


    # SAE as used in Mechanistic Interpretability
    # model = SAE(num_features, num_features * 8, initial_decoder_bias=None)
    # train_sae(model, dataset, epochs=20000, meaning_space=None, force_unit_decoder_norm=True)

    # BitSAE
    bitlinear_kwargs = {
        "weight_measure": "AbsMean",
    }
    model = BitSAE(num_features, num_features * 8, initial_decoder_bias=None, bitlinear_kwargs=bitlinear_kwargs)
    train_sae(model, dataset, epochs=1001, meaning_space=None, force_unit_decoder_norm=False, lr=0.01, l1_coefficient=0.0001)




    exit(0) # old test code below
    num_features = 100
    data = torch.randn(2024, num_features)
    # gm = geometric_median(data)
    # print(
    #     "GM avg l2 distance to all points",
    #     torch.linalg.vector_norm(gm - data, ord=2, dim=-1).mean(),
    # )
    gm = None
    model = SAE(num_features, num_features * 8, initial_decoder_bias=gm)
    train_sae(model, data)


if __name__ == "__main__":
    main()
