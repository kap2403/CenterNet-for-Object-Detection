import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F

# Double convolution block (Conv -> BN -> ReLU) * 2
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Upsample block with an option for bilinear interpolation or transposed convolution
class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(UpSample, self).__init__()
        # Upsampling layer
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # Use transposed convolution if not using bilinear
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        
        # Double convolution after upsampling
        self.conv = DoubleConv(in_ch, out_ch)
        
    def forward(self, x1, x2=None):
        # Apply upsampling to the first input tensor
        x1 = self.up(x1)
        
        if x2 is not None:
            # If a second input is provided, concatenate it with x1 (skip connection)
            x = torch.cat([x2, x1], dim=1)
            
            # Padding to ensure feature maps have the same spatial dimensions
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        else:
            x = x1
        
        return self.conv(x)

# CenterNet model for object detection
class CenterNet(nn.Module):
    def __init__(self, n_classes=1, backbone="resnet18"):
        super(CenterNet, self).__init__()

        # Load pretrained backbone (ResNet model)
        self.backbone = self._get_backbone(backbone)
        
        # Number of channels from the backbone (512 for ResNet18)
        num_ch = 512 if backbone in ["resnet18", "resnet34"] else 2048
        
        # Upsampling blocks for decoder
        self.up1 = UpSample(num_ch, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 256)

        # Output classification map (e.g., object presence)
        self.outc = nn.Conv2d(256, n_classes, kernel_size=1)
        
        # Output regression map (e.g., object bounding box)
        self.outr = nn.Conv2d(256, 2, kernel_size=1)

    def _get_backbone(self, model_name):
        """ Helper function to initialize the backbone model """
        if model_name == "resnet18":
            backbone = torchvision.models.resnet18(pretrained=True)
        elif model_name == "resnet34":
            backbone = torchvision.models.resnet34(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {model_name}")
        
        # Remove the fully connected layers and average pooling
        return nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x):
        # Pass input through the backbone
        x = self.backbone(x)

        # Add positional information (upsampling blocks)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        # Get the output classification map (e.g., object heatmap)
        outc = self.outc(x)
        
        # Get the regression output (e.g., bounding box offset)
        outr = self.outr(x)

        return outc, outr
