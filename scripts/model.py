import torch
import torch.nn as nn
import torch.nn.functional as F
class VGGBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(VGGBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1)
    self.batch1 = nn.BatchNorm2d(out_channels)
    self.dropout1 = nn.Dropout(0.5)
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding = 1)
    self.batch2 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()
    # self.dropout2 = nn.Dropout(0.5)

  def forward(self, x):
    x = self.conv1(x)
    x = self.batch1(x)
    x = self.relu(x)
    x = self.dropout1(x)
    x = self.conv2(x)
    x = self.batch2(x)
    x = self.relu(x)
    # x = self.dropout2(x)
    return x

class VGG16(nn.Module):
    def __init__(self, img_size = 256):
      super(VGG16, self).__init__()
      self.block1 = VGGBlock(3, 64)
      self.maxpool = nn.MaxPool2d(2, 2)
      self.block2 = VGGBlock(64, 128)
      self.block3 = VGGBlock(128, 256)
      self.block4 = VGGBlock(256, 512)
      self.linear1 = nn.Linear(512, 512)
      self.dropout1 = nn.Dropout(0.5)
      self.linear2 = nn.Linear(512, 512)
      self.dropout2 = nn.Dropout(0.5)
      self.linear3 = nn.Linear(512, 1)
      
    def forward(self, x):
      x = self.block1(x)
      x = self.maxpool(x)
      x = self.block2(x)
      x = self.maxpool(x)
      x = self.block3(x)
      x = self.maxpool(x)
      x = self.block4(x)
      x = self.maxpool(x)
      x = x.view(x.size(0), -1)
      # return x
      x = self.linear1(x)
      x = F.relu(x)
      x = self.dropout1(x)
      x = self.linear2(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.linear3(x)
      return x
    
    # @torch.no_grad()
    # def predict(self, x):
    #     self.eval()
    #     x = self.forward(x)
    #     x = F.sigmoid(x)
    #     return x