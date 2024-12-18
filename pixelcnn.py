import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_accuracy_metrics


class PixelCNN(nn.Module):
    def __init__(self, num_channels=10, filters=64, num_layers=7):
        super(PixelCNN, self).__init__()
        self.filters = filters
        self.num_layers = num_layers

        # Initial masked convolution
        # input: (30, 30, 1) -> output: (30, 30, 64)
        self.input_conv = nn.Conv2d(
            1, filters, kernel_size=7, padding=3, bias=False)

        # Gated convolutional layers
        self.conv_layers = nn.ModuleList([nn.Conv2d(
            filters, filters, kernel_size=3, padding=1, bias=False) for _ in range(num_layers)])
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm2d(filters) for _ in range(num_layers)])
        self.relu = nn.ReLU()

        # Output layer with categorical predictions
        self.output_conv = nn.Conv2d(
            filters, num_channels, kernel_size=1, padding=0)

        # Initialize all layers with Kaiming initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def forward(self, x, condition):
    def forward(self, x):
        x = self.input_conv(x)

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)

        x = self.output_conv(x)
        return F.softmax(x, dim=1)


class ConditionalPixelCNN(nn.Module):
    def __init__(self, num_channels=10, filters=64, num_layers=7):
        super(ConditionalPixelCNN, self).__init__()
        self.filters = filters
        self.num_layers = num_layers

        # Initial masked convolution
        # input: (30, 30, 1) -> output: (30, 30, 64)
        self.input_conv = nn.Conv2d(
            1, filters, kernel_size=7, padding=3, bias=False)
        # input: (30, 30, 1) -> output: (30, 30, 64)
        self.cond_conv = nn.Conv2d(
            1, filters, kernel_size=1, padding=0, bias=False)

        # Gated convolutional layers
        self.conv_layers = nn.ModuleList([nn.Conv2d(
            filters, filters, kernel_size=3, padding=1, bias=False) for _ in range(num_layers)])
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm2d(filters) for _ in range(num_layers)])
        self.relu = nn.ReLU()

        # Output layer with categorical predictions
        self.output_conv = nn.Conv2d(
            filters, num_channels, kernel_size=1, padding=0)

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming init"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def forward(self, x, condition):
    def forward(self, x):
        cond = self.cond_conv(x)
        x = self.input_conv(x)
        x = x + cond  # Combine condition with input

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)

        x = self.output_conv(x)
        return F.softmax(x, dim=1)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())
        _, _, h, w = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, h // 2, w // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PixelCNNPlusPlus(nn.Module):
    def __init__(self, input_channels, n_classes, hidden_channels, num_layers=6):
        super().__init__()
        self.layers = nn.Sequential(
            MaskedConv2d('A', input_channels, hidden_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            *[nn.Sequential(
                MaskedConv2d('B', hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU()) for _ in range(num_layers)],
            nn.Conv2d(hidden_channels, n_classes, kernel_size=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming init"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)

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
        metrics["accuracy"] = batch_acc

        return outputs, metrics
