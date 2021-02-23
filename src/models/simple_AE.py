import torch.nn as nn


class SimpleAE(nn.Module):
    def __init__(self, latent_space_dimension):
        super(SimpleAE, self).__init__()
        self.latent_dim = latent_space_dimension

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, self.latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh())

    def encode(self, image):
        x = image.view(image.size(0), -1)
        latent_code = self.encoder(x)
        return latent_code

    def decode(self, code):
        x = self.decoder(code)
        recon_image = x.view(x.size(0), 1, 28, 28)
        return recon_image

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon
