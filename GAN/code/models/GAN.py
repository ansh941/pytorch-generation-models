import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen1 = nn.Linear(100, 300)
        self.gen1_bn = nn.BatchNorm1d(300)
        self.gen2 = nn.Linear(300, 28*28)
        self.gen2_bn = nn.BatchNorm1d(28*28)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

    def get_logits(self, x):
        #gen1 = self.lrelu(self.gen1_bn(self.gen1(x)))
        #gen2 = self.sigmoid(self.gen2_bn(self.gen2(gen1)))
        gen1 = self.lrelu(self.gen1(x))
        gen2 = self.sigmoid(self.gen2(gen1))

        return gen2

    def forward(self, x):
        logits = self.get_logits(x)
        return logits

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dis1 = nn.Linear(28*28, 300)
        self.dis1_bn = nn.BatchNorm1d(300)
        self.dis2 = nn.Linear(300, 2)
        self.dis2_bn = nn.BatchNorm1d(2)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def get_logits(self, x):
        #dis1 = self.lrelu(self.dis1_bn(self.dis1(x)))
        #dis2 = self.lrelu(self.dis2_bn(self.dis2(dis1)))
        dis1 = self.lrelu(self.dis1(x))
        dis2 = self.lrelu(self.dis2(dis1))

        return dis2
    
    def forward(self, x):
        logits = self.get_logits(x)
        return logits
