"""
Coordinate Attention Module
Paper: Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)
Giúp mô hình tập trung vào vị trí quan trọng (chữ khắc, hình dạng viên thuốc)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention cho pill identification
    Tập trung vào đặc trưng không gian của viên thuốc
    """
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        
        # Encode theo chiều height
        x_h = self.pool_h(x)  # (n, c, h, 1)
        
        # Encode theo chiều width
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (n, c, w, 1)
        
        # Concatenate và encode
        y = torch.cat([x_h, x_w], dim=2)  # (n, c, h+w, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split lại
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # Attention weights
        a_h = self.conv_h(x_h).sigmoid()  # (n, c, h, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (n, c, 1, w)
        
        # Apply attention
        out = identity * a_h * a_w
        
        return out


if __name__ == '__main__':
    # Test module
    x = torch.randn(4, 256, 14, 14)
    ca = CoordinateAttention(in_channels=256, reduction=32)
    out = ca(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert x.shape == out.shape, "Shape mismatch!"
    print("✓ Coordinate Attention test passed!")