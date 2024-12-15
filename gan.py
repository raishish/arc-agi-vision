
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, embedding_dim=32, img_size=30):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, img_size * img_size),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, noise, labels):
        # Embed labels and concatenate with noise
        label_embedding = self.embedding(labels)
        gen_input = torch.cat([noise, label_embedding], dim=1)
        img = self.model(gen_input)
        return img.view(img.size(0), 1, self.img_size, self.img_size)


class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_size=30, embedding_dim=32):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(1 * img_size * img_size + embedding_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, img, labels):
        # Embed labels and concatenate with flattened image
        label_embedding = self.embedding(labels)
        img_flat = img.view(img.size(0), -1)
        disc_input = torch.cat([img_flat, label_embedding], dim=1)
        return self.model(disc_input)


def train_model(
    model,
    dataloader,
    epochs,
    optimizer,
    criterion,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    # criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in dataloader:  # Inputs: (B, 1, 30, 30), Targets: (B, 30, 30)
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # Outputs: (B, 10, 30, 30)
            loss = criterion(outputs.permute(0, 2, 3, 1).reshape(-1, 10), targets.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
