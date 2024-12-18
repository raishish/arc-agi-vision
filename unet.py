import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_accuracy_metrics


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upconv(x)
        # Pad to handle odd and even dimensions
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]])
        x = torch.cat([x, skip], dim=1)  # Concatenate skip connections
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, base_channels=64, num_blocks=3):
        super(UNet, self).__init__()
        # Encoder blocks
        self.enc_blocks = []
        hidden_channels = base_channels
        for block in range(num_blocks):
            self.enc_blocks.append(EncoderBlock(in_channels, hidden_channels))
            in_channels = hidden_channels
            hidden_channels *= 2

        # dividing by 2 to account for 2x in channels in the last for loop iteration above (after break)
        hidden_channels //= 2

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True)
        )

        # Decoder blocks
        self.dec_blocks = []
        for block in range(num_blocks):
            self.dec_blocks.append(DecoderBlock(hidden_channels * 2, hidden_channels))
            hidden_channels //= 2

        # Output Layer
        self.output_layer = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skips = []
        for enc_block in self.enc_blocks:
            skip, x = enc_block(x)
            skips.append(skip)

        skips.reverse()  # reverse skips to match the dimensions in the DecoderBlock

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for dec_block, skip in zip(self.dec_blocks, skips):
            x = dec_block(x, skip)

        # Output
        return torch.softmax(self.output_layer(x), dim=1)

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
            data (torch.Tensor): tuple of (input, target) tensors
            batch_num (int): batch number
            model (torch.nn.Module): model to use
            device (torch.device): device to use
            mode (str): mode to use (train, eval, or test)
            optimizer (torch.optim.Optimizer): optimizer to use
            loss_criterion (torch.nn.Module): loss criterion to use
            acc_criterion (torch.nn.Module, optional): accuracy criterion to use
            verbose (int, optional): verbosity level (0, 1, or 2)

        Returns:
            dict: metrics for the batch (loss, accuracy)
        """
        # Move data to the device
        inputs, targets = data[0].to(device), data[1].to(device).squeeze()

        # Forward pass
        outputs = self.forward(inputs)

        # Calculate loss
        loss = loss_criterion(outputs, targets.long())

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
        metrics["accuracy"] = batch_acc.item()

        return outputs, metrics
