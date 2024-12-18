import torch
import torch.nn as nn
from utils import get_accuracy_metrics


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
            nn.Unflatten(1, (num_classes, h, w)),
            nn.Softmax2d()
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

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

    def process_one_batch(
        self,
        data: tuple,
        optimizer: torch.optim.Optimizer,
        loss_criterion: torch.nn.Module,
        acc_criterion=None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        mode: str = "eval",
    ):
        """
        Processes one batch of data.

        Args:
            data: tuple of (input, target) tensors
            optimizer: optimizer
            loss_criterion: loss criterion
            acc_criterion (optional): accuracy criterion
            device (torch.device): device
            mode (str): mode to use (train, eval, or test)

        Returns:
            tuple: (outputs, metrics)
        """
        # Move data to the device
        inputs, targets = data[0].to(device), data[1].to(device).squeeze()

        # Forward pass
        outputs, mu, logvar = self.forward(inputs)

        # Calculate loss
        recon_loss = loss_criterion(outputs, targets.long())
        kl_loss = KL_Loss(mu, logvar)
        loss = recon_loss + kl_loss

        # Backpropagate
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate accuracy
        if acc_criterion:
            batch_acc = acc_criterion(outputs, targets)
        else:
            batch_acc = None

        metrics = get_accuracy_metrics(outputs, targets)
        metrics["loss"] = loss.item()
        metrics["reconstruction_loss"] = recon_loss.item()
        metrics["kl_loss"] = kl_loss.item()
        metrics["accuracy"] = batch_acc.item()

        return outputs, metrics


def KL_Loss(mu, logvar):
    """KL Divergence loss"""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
