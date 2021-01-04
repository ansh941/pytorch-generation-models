import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen1 = nn.Linear(100+10, 16*16*96)
        self.gen1_bn = nn.BatchNorm1d(16*16*96)

        self.deconv1 = nn.ConvTranspose2d(96, 64, 5)
        self.deconv1_bn = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 5)
        self.deconv2_bn = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 5)
        self.deconv3_bn = nn.BatchNorm2d(1)
        
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

    def get_logits(self, x):
        gen1 = self.lrelu(self.gen1_bn(self.gen1(x))).view(-1,96,16,16)
        
        deconv1 = self.lrelu(self.deconv1_bn(self.deconv1(gen1)))
        deconv2 = self.lrelu(self.deconv2_bn(self.deconv2(deconv1)))
        deconv3 = self.sigmoid(self.deconv3_bn(self.deconv3(deconv2)))
        
        return deconv3

    def forward(self, x,  y):
        x = torch.cat((x,y), dim=1)
        logits = self.get_logits(x)
        return logits

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(11, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, 5)
        self.conv3_bn = nn.BatchNorm2d(96)

        self.dis1 = nn.Linear(16*16*96, 2)
        self.dis1_bn = nn.BatchNorm1d(2)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.cls1 = nn.Linear(16*16*96, 10)
        self.cls1_bn = nn.BatchNorm1d(10)

    def get_logits(self, x):
        conv1 = self.lrelu(self.conv1_bn(self.conv1(x)))
        conv2 = self.lrelu(self.conv2_bn(self.conv2(conv1)))
        conv3 = self.lrelu(self.conv3_bn(self.conv3(conv2)))
        conv3 = conv3.view(-1, 96*16*16)

        dis1 = self.lrelu(self.dis1(conv3))
        cls1 = self.lrelu(self.cls1(conv3))

        return dis1, cls1
    
    def conv_cond_concat(self, x, y):
        y = y.view(-1, 10, 1, 1)
        new_y = y*torch.ones((x.size(0), y.size(1), x.size(2), x.size(3))).cuda()
        return torch.cat((x, new_y), dim=1)
    
    def forward(self, x, y):
        x = self.conv_cond_concat(x, y)
        dis_logits, cls_logits= self.get_logits(x)

        return dis_logits, cls_logits
