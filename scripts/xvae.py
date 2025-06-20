"""PyTorch implementation of the X-shaped variational autoencoder used for X-DEC."""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.common import sse, bce, mmd, sampling, kl_regu


def _get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    return nn.ELU()


class xvae(nn.Module):
    """X-VAE integrating two data modalities."""

    def __init__(
        self,
        s1_input_size,
        s2_input_size,
        ds1=48,
        ds2=None,
        ds12=None,
        ls=32,
        weighted=True,
        act="elu",
        dropout=0.2,
        distance="kl",
        beta=25,
        epochs=250,
        bs=64,
        save_model=False,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.s1_input_size = s1_input_size
        self.s2_input_size = s2_input_size
        self.ds1 = ds1
        self.ds2 = ds2 if ds2 is not None else ds1
        self.ds12 = ds12 if ds12 is not None else ds1
        self.ls = ls
        self.weighted = weighted
        self.act_fn = _get_activation(act)
        self.dropout_rate = dropout
        self.distance = distance
        self.beta = beta
        self.epochs = epochs
        self.bs = bs
        self.save_model_flag = save_model

        # Encoder
        self.enc_fc1_s1 = nn.Linear(s1_input_size, self.ds1)
        self.enc_bn1_s1 = nn.BatchNorm1d(self.ds1)
        self.enc_fc1_s2 = nn.Linear(s2_input_size, self.ds2)
        self.enc_bn1_s2 = nn.BatchNorm1d(self.ds2)
        self.enc_fc2 = nn.Linear(self.ds1 + self.ds2, self.ds12)
        self.enc_bn2 = nn.BatchNorm1d(self.ds12)
        self.z_mean = nn.Linear(self.ds12, self.ls)
        self.z_log_sigma = nn.Linear(self.ds12, self.ls)

        # Decoder
        self.dec_fc = nn.Linear(self.ls, self.ds12)
        self.dec_bn = nn.BatchNorm1d(self.ds12)
        self.dec_fc_s1 = nn.Linear(self.ds12, self.ds1)
        self.dec_bn_s1 = nn.BatchNorm1d(self.ds1)
        self.dec_fc_s2 = nn.Linear(self.ds12, self.ds2)
        self.dec_bn_s2 = nn.BatchNorm1d(self.ds2)
        self.out_s1 = nn.Linear(self.ds1, s1_input_size)
        self.out_s2 = nn.Linear(self.ds2, s2_input_size)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.to(self.device)

    # ------------------------------------------------------------------
    # Model components
    def encode(self, s1, s2):
        x1 = self.act_fn(self.enc_bn1_s1(self.enc_fc1_s1(s1)))
        x2 = self.act_fn(self.enc_bn1_s2(self.enc_fc1_s2(s2)))
        x = torch.cat([x1, x2], dim=1)
        x = self.act_fn(self.enc_bn2(self.enc_fc2(x)))
        mean = self.z_mean(x)
        log_sigma = self.z_log_sigma(x)
        return mean, log_sigma

    def decode(self, z):
        x = self.act_fn(self.dec_bn(self.dec_fc(z)))
        x = self.dropout(x)
        x1 = self.act_fn(self.dec_bn_s1(self.dec_fc_s1(x)))
        x2 = self.act_fn(self.dec_bn_s2(self.dec_fc_s2(x)))
        s1_out = self.out_s1(x1)
        s2_out = torch.sigmoid(self.out_s2(x2))
        return s1_out, s2_out

    def forward(self, s1, s2):
        mean, log_sigma = self.encode(s1, s2)
        z = sampling((mean, log_sigma))
        s1_out, s2_out = self.decode(z)
        return s1_out, s2_out, mean, log_sigma, z

    # ------------------------------------------------------------------
    def fit(self, s1_train, s2_train, s1_val=None, s2_val=None):
        dataset = TensorDataset(
            torch.tensor(s1_train, dtype=torch.float32),
            torch.tensor(s2_train, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.bs, shuffle=True)

        optim = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-3)
        self.train()
        for _ in range(self.epochs):
            for b1, b2 in loader:
                b1 = b1.to(self.device)
                b2 = b2.to(self.device)
                optim.zero_grad()
                s1_out, s2_out, mean, log_sigma, z = self.forward(b1, b2)
                if self.weighted:
                    l1 = sse(b1, s1_out) * self.s1_input_size
                    l2 = bce(b2, s2_out) * self.s2_input_size
                else:
                    l1 = sse(b1, s1_out)
                    l2 = bce(b2, s2_out)
                rec = l1 + l2
                if self.weighted:
                    rec = rec / (self.s1_input_size + self.s2_input_size)
                if self.distance == "mmd":
                    prior = torch.randn_like(z)
                    dist = mmd(prior, z)
                else:
                    dist = kl_regu(mean, log_sigma)
                loss = torch.mean(rec + self.beta * dist)
                loss.backward()
                optim.step()

        if self.save_model_flag:
            self.save_model()

    def predict(self, s1_data, s2_data, output="encoder"):
        self.eval()
        with torch.no_grad():
            s1 = torch.tensor(s1_data, dtype=torch.float32, device=self.device)
            s2 = torch.tensor(s2_data, dtype=torch.float32, device=self.device)
            s1_out, s2_out, mean, log_sigma, _ = self.forward(s1, s2)
            if output == "encoder":
                return mean.cpu().numpy()
            return (s1_out.cpu().numpy(), s2_out.cpu().numpy())

    # ------------------------------------------------------------------
    def save_model(self, path="xvae_model", force_path=True):
        if force_path and not os.path.exists(path):
            os.makedirs(path)
        torch.save({"state_dict": self.state_dict(), "params": self._save_params()}, os.path.join(path, "model.pt"))

    def _save_params(self):
        return {
            "s1_input_size": self.s1_input_size,
            "s2_input_size": self.s2_input_size,
            "ds1": self.ds1,
            "ds2": self.ds2,
            "ds12": self.ds12,
            "ls": self.ls,
            "weighted": self.weighted,
            "act": type(self.act_fn).__name__,
            "dropout": self.dropout_rate,
            "distance": self.distance,
            "beta": self.beta,
            "epochs": self.epochs,
            "bs": self.bs,
        }


def load_xvae_model(path, device="cpu"):
    chk = torch.load(os.path.join(path, "model.pt"), map_location=device)
    params = chk["params"]
    act_map = {"ReLU": "relu", "Tanh": "tanh", "ELU": "elu"}
    act = act_map.get(params.get("act", "ELU"), "elu")
    model = xvae(
        params["s1_input_size"],
        params["s2_input_size"],
        ds1=params["ds1"],
        ds2=params["ds2"],
        ds12=params["ds12"],
        ls=params["ls"],
        weighted=params["weighted"],
        act=act,
        dropout=params["dropout"],
        distance=params["distance"],
        beta=params["beta"],
        epochs=params["epochs"],
        bs=params["bs"],
        device=device,
    )
    model.load_state_dict(chk["state_dict"])
    model.to(device)
    return model
