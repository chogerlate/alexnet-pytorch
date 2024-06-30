"""
Contains PyTorch model code to instantiate a Alexnet model.
"""

import torch
from torch import nn


class AlexNet(nn.Module):
    """Creates an AlexNet model

    A little bit of interesting implementation details:
        - The original AlexNet model was trained on the ImageNet dataset.
        - LRN layers are used in the original AlexNet model, but they are not used in modern implementations.
        - we can replace the LRN layers with BatchNorm2d layers.
        - We can add AdaptiveAvgPool2d layer before classifier layer to make it more modern.
    Args:
        num_classes (int): class numbers of the dataset. Default is 1000.
    """

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.num_classes = num_classes
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0
            ),  #  -> (b, 3, 55, 55)
            nn.ReLU(),  # -> (b, 96, 55, 55)
            nn.LocalResponseNorm(
                alpha=1e-4, beta=0.75, k=2, size=5
            ),  # apply LRN -> (b, 96, 27, 27)
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> (b, 96, 27, 27)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, padding=2
            ),  # -> (b, 256, 27, 27)
            nn.ReLU(),  # -> (b, 256, 13, 13 )
            nn.LocalResponseNorm(
                alpha=1e-4, beta=0.75, k=2, size=5
            ),  # apply LRN -> (b, 256, 13, 13)
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> (b, 256, 13, 13)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=384, kernel_size=3, padding=1
            ),  # -> (b, 384, 13, 13)
            nn.ReLU(),  # -> (b, 384, 13, 13 )
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=384, out_channels=384, kernel_size=3, padding=1
            ),  # -> (b, 384, 13, 13)
            nn.ReLU(),  # -> (b, , 384, 13, 13)
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(
                in_channels=384, out_channels=256, kernel_size=3, padding=1
            ),  # -> (b, 256, 13, 13)
            nn.ReLU(),  # -> (b, 256, 13, 13)
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> (b, 256, 6, 6)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),  # -> (b, 4096)
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),  # -> (b, 4096)
            nn.ReLU(),
            nn.Linear(
                in_features=4096, out_features=self.num_classes
            ),  # -> (b, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights of the model following the original paper.
        Args:
            module ([nn.Module]): [nn.Module to initialize weights.]
        """
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                if module in [
                    self.conv_block_2[0],
                    self.conv_block_4[0],
                    self.conv_block_5[0],
                ]:
                    nn.init.constant_(module.bias, 1)
                else:
                    nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.constant_(module.bias, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
