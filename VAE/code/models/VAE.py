import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, n_gaussians):
        super(VAE, self).__init__()
        self.encoder = Encoder(n_gaussians)
        self.decoder = Decoder(n_gaussians)

    def forward(self, x):
        mu, logvar, z = self.encoder(x)
        decoded = self.decoder(z)

        return decoded, mu, logvar

class Encoder(nn.Module):
    def __init__(self, n_gaussians):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, bias=False)    # output becomes 26x26
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 7, bias=False)   # output becomes 20x20
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 7, bias=False)  # output becomes 14x14
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 7, bias=False) # output becomes 8x8
        self.conv4_bn = nn.BatchNorm2d(256)
        self.mu = nn.Linear(256*8*8, n_gaussians, bias=False)
        self.mu_bn = nn.BatchNorm1d(n_gaussians)
        self.logvar = nn.Linear(256*8*8, n_gaussians, bias=False)
        self.logvar_bn = nn.BatchNorm1d(n_gaussians)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def encode(self, x):
        conv1 = self.lrelu(self.conv1_bn(self.conv1(x)))
        conv2 = self.lrelu(self.conv2_bn(self.conv2(conv1)))
        conv3 = self.lrelu(self.conv3_bn(self.conv3(conv2)))
        conv4 = self.lrelu(self.conv4_bn(self.conv4(conv3)))
        flat1 = torch.flatten(conv4.permute(0, 2, 3, 1), 1)

        mu = self.mu_bn(self.mu(flat1))
        logvar = self.logvar_bn(self.logvar(flat1))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(torch.mul(logvar,0.5))
        eps = torch.autograd.Variable(torch.randn(*std.size())).cuda()
        return eps*std+mu
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

class Decoder(nn.Module):
    def __init__(self, n_gaussians):
        super(Decoder, self).__init__()
        self.z = nn.Linear(n_gaussians, 256*8*8)
            
        self.deconv1 = nn.ConvTranspose2d(256, 128, 7, bias=False)
        self.deconv1_bn = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 7, bias=False)
        self.deconv2_bn = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 7, bias=False)
        self.deconv3_bn = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 7, bias=False)
        self.deconv4_bn = nn.BatchNorm2d(3)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def decode(self, x):
        z = self.z(x)
        z = z.view(-1, 256, 8, 8)
        deconv1 = self.lrelu(self.deconv1_bn(self.deconv1(z)))
        deconv2 = self.lrelu(self.deconv2_bn(self.deconv2(deconv1)))
        deconv3 = self.lrelu(self.deconv3_bn(self.deconv3(deconv2)))
        deconv4 = self.deconv4_bn(self.deconv4(deconv3))
        decoded = self.tanh(deconv4)
        return (decoded+1)/2

    def forward(self, x):
        decoded = self.decode(x)
        return decoded
