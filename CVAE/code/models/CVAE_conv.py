import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, n_gaussians):
        super(CVAE, self).__init__()
        self.encoder = Encoder(n_gaussians)
        self.decoder = Decoder(n_gaussians)

    def forward(self, x, y):
        mu, logvar, z = self.encoder(x, y)
        decoded = self.decoder(z, y)

        return decoded, mu, logvar

class Encoder(nn.Module):
    def __init__(self, n_gaussians):
        super(Encoder, self).__init__()
        self.n_gaussians = n_gaussians

        self.conv1 = nn.Conv2d(11, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, 5)
        self.conv3_bn = nn.BatchNorm2d(96)

        self.dis1 = nn.Linear(16*16*96, 2)
        self.dis1_bn = nn.BatchNorm1d(2)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.mu = nn.Linear(16*16*96, n_gaussians, bias=False)
        self.mu_bn = nn.BatchNorm1d(n_gaussians)
        self.logvar = nn.Linear(16*16*96, n_gaussians, bias=False)
        self.logvar_bn = nn.BatchNorm1d(n_gaussians)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def encode(self, x):
        conv1 = self.lrelu(self.conv1_bn(self.conv1(x)))
        conv2 = self.lrelu(self.conv2_bn(self.conv2(conv1)))
        conv3 = self.lrelu(self.conv3_bn(self.conv3(conv2)))
        flat1 = conv3.view(-1, 96*16*16)

        mu = self.mu_bn(self.mu(flat1))
        logvar = self.logvar_bn(self.logvar(flat1))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(torch.mul(logvar,0.5))
        eps = torch.randn(*std.size()).cuda()
        eps = eps*std+mu
        eps = torch.autograd.Variable(eps)
        return eps

    def conv_cond_concat(self, x, y):
        y = y.view(-1, 10, 1, 1)
        new_y = y*torch.ones((x.size(0), y.size(1), x.size(2), x.size(3))).cuda()
        return torch.cat((x, new_y), dim=1)

    def forward(self, x, y):
        x = self.conv_cond_concat(x,y)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

class Decoder(nn.Module):
    def __init__(self, n_gaussians):
        super(Decoder, self).__init__()
        self.n_gaussians = n_gaussians
        self.z1 = nn.Linear(100+10, 16*16*96)
        self.z1_bn = nn.BatchNorm1d(16*16*96)

        self.deconv1 = nn.ConvTranspose2d(96, 64, 5)
        self.deconv1_bn = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 5)
        self.deconv2_bn = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 5)
        self.deconv3_bn = nn.BatchNorm2d(1)

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def decode(self, x):
        z1 = self.lrelu(self.z1_bn(self.z1(x))).view(-1,96,16,16)

        deconv1 = self.lrelu(self.deconv1_bn(self.deconv1(z1)))
        deconv2 = self.lrelu(self.deconv2_bn(self.deconv2(deconv1)))
        deconv3 = self.sigmoid(self.deconv3_bn(self.deconv3(deconv2)))

        return deconv3

    def forward(self, x, y):
        x = torch.cat((x,y), dim=1)
        decoded = self.decode(x)
        return decoded
