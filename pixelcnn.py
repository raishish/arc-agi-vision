# import torch
import torch.nn as nn
import torch.nn.functional as F


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
