import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, 5)
        self.conv3_bn = nn.BatchNorm2d(96)

        self.z1 = nn.Linear(16*16*96, 100)
        self.z1_bn = nn.BatchNorm1d(100)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def get_logits(self, x):
        conv1 = self.lrelu(self.conv1_bn(self.conv1(x)))
        conv2 = self.lrelu(self.conv2_bn(self.conv2(conv1)))
        conv3 = self.lrelu(self.conv3_bn(self.conv3(conv2)))
        conv3 = conv3.view(-1, 96*16*16)
        z1 = self.lrelu(self.z1(conv3))

        return z1
    
    def forward(self, x):
        logits = self.get_logits(x)
        return logits

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.z1 = nn.Linear(100, 16*16*96)
        self.z1_bn = nn.BatchNorm1d(16*16*96)

        self.deconv1 = nn.ConvTranspose2d(96, 64, 5)
        self.deconv1_bn = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 5)
        self.deconv2_bn = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 5)
        self.deconv3_bn = nn.BatchNorm2d(1)
        
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

    def get_logits(self, x):
        z1 = self.lrelu(self.z1_bn(self.z1(x))).view(-1,96,16,16)
        
        deconv1 = self.lrelu(self.deconv1_bn(self.deconv1(z1)))
        deconv2 = self.lrelu(self.deconv2_bn(self.deconv2(deconv1)))
        deconv3 = self.sigmoid(self.deconv3_bn(self.deconv3(deconv2)))
        
        return deconv3

    def forward(self, x):
        logits = self.get_logits(x)
        return logits

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis1 = nn.Linear(100, 50)
        self.dis1_bn = nn.BatchNorm1d(50)
        self.dis2 = nn.Linear(50, 2)
        self.dis2_bn = nn.BatchNorm1d(2)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def get_logits(self, x):
        dis1 = self.lrelu(self.dis1_bn(self.dis1(x)))
        dis2 = self.lrelu(self.dis2_bn(self.dis2(dis1)))

        return dis2
    
    def forward(self, x):
        logits = self.get_logits(x)
        return logits
