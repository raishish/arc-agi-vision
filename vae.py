import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim=30*30,
        latent_dim=20,
        hidden_dim=32
    ):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        hidden = self.fc(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim=20,
        output_dim=30*30,
        num_classes=10,
        hidden_dim=32
    ):
        super(Decoder, self).__init__()
        h, w = int(output_dim ** 0.5), int(output_dim ** 0.5)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes * output_dim),
            nn.ReLU(),
            nn.Unflatten(1, (num_classes, h, w))
        )

    def forward(self, z):
        x_reconstructed = self.fc(z)
        return x_reconstructed


class VAE(nn.Module):
    def __init__(
        self,
        input_dim=30*30,
        latent_dim=20,
        num_classes=10,
        hidden_dim=32
    ):
        super(VAE, self).__init__()
        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        """
        return_logits: return logits if True
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)

        probs = torch.softmax(logits, dim=1)
        if return_logits:
            logits, probs, mu, logvar

        return logits, mu, logvar
