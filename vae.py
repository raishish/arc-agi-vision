import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim=30*30, latent_dim=20, num_classes=10, embedding_dim=32):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.fc1 = nn.Linear(input_dim + embedding_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)  # Mean vector
        self.fc_logvar = nn.Linear(256, latent_dim)  # Log variance vector

    def forward(self, x, labels):
        label_embedding = self.embedding(labels)  # Shape: (batch_size, embedding_dim)
        x = x.view(x.size(0), -1)  # Flatten the image
        x = torch.cat([x, label_embedding], dim=1)  # Concatenate image and label
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=20, output_dim=30*30, num_classes=10, embedding_dim=32):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.fc = nn.Linear(latent_dim + embedding_dim, 256)
        self.output_layer = nn.Linear(256, output_dim)

    def forward(self, z, labels):
        label_embedding = self.embedding(labels)  # Shape: (batch_size, embedding_dim)
        z = torch.cat([z, label_embedding], dim=1)  # Concatenate latent vector and label
        h = torch.relu(self.fc(z))
        x_reconstructed = torch.sigmoid(self.output_layer(h))  # Reconstruct the image
        return x_reconstructed


class CVAE(nn.Module):
    def __init__(self, input_dim=30*30, latent_dim=20, num_classes=10, embedding_dim=32):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, num_classes, embedding_dim)
        self.decoder = Decoder(latent_dim, input_dim, num_classes, embedding_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z, labels)
        return x_reconstructed, mu, logvar


def cvae_loss(reconstructed, original, mu, logvar):
    # Reconstruction loss
    recon_loss = nn.BCELoss(reduction='sum')(reconstructed, original)

    # KL Divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss


def train_model(
    model,
    dataloader,
    epochs,
    optimizer,
    device
):
    # Initialize CVAE, optimizer, and data
    cvae = model.to(device)

    # Training loop
    for epoch in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            reconstructed, mu, logvar = cvae(images, labels)
            loss = cvae_loss(reconstructed, images.view(images.size(0), -1), mu, logvar)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item() / len(images):.4f}")
