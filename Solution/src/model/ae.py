import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class VAE(nn.Module):
    def __init__(self, dim_input, z_dim=32):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.dim_input = dim_input
        self.encoder = nn.Sequential(
            nn.Linear(dim_input, dim_input//2),
            nn.ReLU(True),
            nn.Linear(dim_input//2, dim_input//4),
        )
        self.fc_mu = nn.Linear(dim_input//4, z_dim)
        self.fc_logvar = nn.Linear(dim_input//4, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, dim_input//4),
            nn.ReLU(True),
            nn.Linear(dim_input//4, dim_input//2),
            nn.ReLU(True),
            nn.Linear(dim_input//2, dim_input),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)
        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def loss(self, x, recon, mu, logvar, beta):
        mse_loss = nn.MSELoss()
        MSE = mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD


class AE(nn.Module):
    def __init__(self, dim_input_list, z_dim=32):
        super(AE, self).__init__()
        self.z_dim = z_dim
        self.dim_input = dim_input
        self.encoder = nn.Sequential(
            nn.Linear(dim_input_list[0], dim_input_list[1]),
            nn.ReLU(True),
            nn.Linear(dim_input_list[1], dim_input_list[2]),
        )
        self.fc_z = nn.Linear(dim_input_list[2], z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, dim_input_list[2]),
            nn.ReLU(True),
            nn.Linear(dim_input_list[2], dim_input_list[1]),
            nn.ReLU(True),
            nn.Linear(dim_input_list[1], dim_input_list[0]),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        z = self._encode(x)
        z_ = self.fc_z(z)
        x_recon = self._decode(z_)
        return x_recon, z_

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def loss(self, x, recon):
        mse_loss = nn.MSELoss()
        MSE = mse_loss(recon, x)
        return MSE
