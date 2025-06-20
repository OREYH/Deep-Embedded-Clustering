import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans


class AutoEncoder(nn.Module):
    """Simple MLP autoencoder used for DEC."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


class DECLayer(nn.Module):
    """Clustering layer using Student's t-distribution."""

    def __init__(self, n_clusters: int, latent_dim: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, z):
        q = 1.0 / (1.0 + (torch.sum((z.unsqueeze(1) - self.clusters) ** 2, dim=2) / self.alpha))
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q


def target_distribution(q):
    weight = (q ** 2) / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()


def train_dec(data: np.ndarray, n_clusters: int, hidden_dim: int = 64,
              latent_dim: int = 8, batch_size: int = 256, pretrain_epochs: int = 50,
              dec_epochs: int = 100, lr: float = 1e-3, device: str = 'cpu'):
    """Train a DEC model on the given data."""
    device = torch.device(device)
    x = torch.tensor(data, dtype=torch.float32)
    dataset = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)

    model = AutoEncoder(data.shape[1], hidden_dim, latent_dim).to(device)
    optim_ae = torch.optim.Adam(model.parameters(), lr=lr)

    # pretrain autoencoder
    model.train()
    for _ in range(pretrain_epochs):
        for (batch,) in dataset:
            batch = batch.to(device)
            optim_ae.zero_grad()
            recon = model(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            optim_ae.step()

    # obtain latent features for k-means
    with torch.no_grad():
        z = model.encode(x.to(device)).cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z)

    dec_layer = DECLayer(n_clusters, latent_dim).to(device)
    dec_layer.clusters.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    params = list(model.parameters()) + list(dec_layer.parameters())
    optim_dec = torch.optim.Adam(params, lr=lr)

    for _ in range(dec_epochs):
        for (batch,) in dataset:
            batch = batch.to(device)
            z = model.encode(batch)
            recon = model.decoder(z)
            q = dec_layer(z)
            p = target_distribution(q).detach()
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            recon_loss = F.mse_loss(recon, batch)
            loss = recon_loss + kl_loss
            optim_dec.zero_grad()
            loss.backward()
            optim_dec.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(x.to(device))
        q = dec_layer(z)
    final_pred = q.argmax(dim=1).cpu().numpy()
    return final_pred, z.cpu().numpy(), model, dec_layer


def save_dec_model(model: AutoEncoder, dec_layer: DECLayer, path: str) -> None:
    """Save a DEC autoencoder and clustering layer."""
    torch.save(
        {
            "ae_state": model.state_dict(),
            "dec_state": dec_layer.state_dict(),
            "params": {
                "input_dim": next(model.parameters()).shape[1],
                "hidden_dim": model.encoder[0].out_features,
                "latent_dim": model.encoder[-1].out_features,
                "n_clusters": dec_layer.clusters.size(0),
            },
        },
        path,
    )


def load_dec_model(path: str, device: str = "cpu"):
    """Load a DEC model saved with ``save_dec_model``."""
    chk = torch.load(path, map_location=device)
    params = chk["params"]
    model = AutoEncoder(
        params["input_dim"], params["hidden_dim"], params["latent_dim"]
    ).to(device)
    dec_layer = DECLayer(params["n_clusters"], params["latent_dim"]).to(device)
    model.load_state_dict(chk["ae_state"])
    dec_layer.load_state_dict(chk["dec_state"])
    model.eval()
    dec_layer.eval()
    return model, dec_layer
