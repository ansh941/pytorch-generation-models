import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, n_gaussians):
        super(CVAE, self).__init__()
        self.encoder = Encoder(n_gaussians)
        self.decoder = Decoder(n_gaussians)

    def forward(self, x, y):
        y = y.view(-1, 1)
        mu, logvar, z = self.encoder(x, y)
        decoded = self.decoder(z, y)

        return decoded, mu, logvar

class Encoder(nn.Module):
    def __init__(self, n_gaussians):
        super(Encoder, self).__init__()
        self.n_gaussians = n_gaussians
        self.fc1 = nn.Linear(28*28+1, 300, bias=False)
        self.fc1_bn = nn.BatchNorm1d(300)
        self.mu = nn.Linear(300, n_gaussians, bias=False)
        self.mu_bn = nn.BatchNorm1d(n_gaussians)
        self.logvar = nn.Linear(300, n_gaussians, bias=False)
        self.logvar_bn = nn.BatchNorm1d(n_gaussians)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def encode(self, x, y):
        x = torch.cat((x,y), dim=1)
        fc1 = self.fc1_bn(self.fc1(x))
        mu = self.mu_bn(self.mu(fc1))
        logvar = self.logvar_bn(self.logvar(fc1))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(torch.mul(logvar,0.5))
        eps = torch.randn(*std.size()).cuda()
        eps = eps*std+mu
        eps = torch.autograd.Variable(eps)
        return eps
    
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

class Decoder(nn.Module):
    def __init__(self, n_gaussians):
        super(Decoder, self).__init__()
        self.n_gaussians = n_gaussians

        self.z1 = nn.Linear(n_gaussians+1, 300)
        self.z1_bn = nn.BatchNorm1d(300)
        self.out = nn.Linear(300, 28*28)
        self.out_bn = nn.BatchNorm1d(28*28)
            
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def decode(self, x):
        z1 = self.lrelu(self.z1_bn(self.z1(x)))
        out = self.out_bn(self.out(z1))

        decoded = self.tanh(out)
        #decoded = self.sigmoid(deconv4)
        return (decoded+1)/2

    def forward(self, x, y):
        if(x.size()[1] == self.n_gaussians):
            x = torch.cat((x, y), dim=1)
        decoded = self.decode(x)
        return decoded
