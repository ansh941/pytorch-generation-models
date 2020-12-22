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
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)         
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv1_relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)       
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv2_relu = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2)                                    # output becomes 16x16
        self.subconv1 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.subconv1_bn = nn.BatchNorm2d(128)
        self.subconv1_relu = nn.ReLU(True)
        self.subconv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.subconv2_bn = nn.BatchNorm2d(128)
        self.subconv2_relu = nn.ReLU(True)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=False)      
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv3_relu = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2)                                    # output becomes 8x8
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1, bias=False)      
        self.conv4_bn = nn.BatchNorm2d(512)
        self.conv4_relu = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(2)                                    # output becomes 4x4
        self.subconv3 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.subconv3_bn = nn.BatchNorm2d(512)
        self.subconv3_relu = nn.ReLU(True)
        self.subconv4 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.subconv4_bn = nn.BatchNorm2d(512)
        self.subconv4_relu = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d(4)                                    # output becomes 1x1
        self.mu = nn.Linear(512, n_gaussians, bias=False)
        self.mu_bn = nn.BatchNorm1d(n_gaussians)
        self.logvar = nn.Linear(512, n_gaussians, bias=False)
        self.logvar_bn = nn.BatchNorm1d(n_gaussians)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def encode(self, x):
        conv1 = self.conv1_relu(self.conv1_bn(self.conv1(x)))
        conv2 = self.conv2_relu(self.conv2_bn(self.conv2(conv1)))
        pool1 = self.pool1(conv2)
        subconv1 = self.subconv1_relu(self.subconv1_bn(self.subconv1(pool1)))
        subconv2 = self.subconv2_relu(self.subconv2_bn(self.subconv2(subconv1)))
        add1 = pool1 + subconv2
        conv3 = self.conv3_relu(self.conv3_bn(self.conv3(add1)))
        pool2 = self.pool2(conv3)
        conv4 = self.conv4_relu(self.conv4_bn(self.conv4(pool2)))
        pool3 = self.pool3(conv4)
        subconv3 = self.subconv3_relu(self.subconv3_bn(self.subconv3(pool3)))
        subconv4 = self.subconv4_relu(self.subconv4_bn(self.subconv4(subconv3)))
        add2 = pool3 + subconv4
        pool4 = self.pool4(add2)
        flat1 = torch.flatten(pool4.permute(0, 2, 3, 1), 1)

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
        self.z = nn.Linear(n_gaussians, 512)
            
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

        self.up1 = nn.Upsample(4)
        self.deconv1 = nn.ConvTranspose2d(512, 256, 3, padding=1, bias=False)
        self.deconv1_bn = nn.BatchNorm2d(256)
        self.up2 = nn.Upsample(8)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 3, padding=1, bias=False)
        self.deconv2_bn = nn.BatchNorm2d(128)
        self.up3 = nn.Upsample(16)

        self.subdeconv1 = nn.ConvTranspose2d(128, 128, 3, padding=1, bias=False)
        self.subdeconv1_bn = nn.BatchNorm2d(128)
        self.subdeconv2 = nn.ConvTranspose2d(128, 128, 3, padding=1, bias=False)
        self.subdeconv2_bn = nn.BatchNorm2d(128)
        
        self.up4 = nn.Upsample(32)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 3, padding=1, bias=False)
        self.deconv3_bn = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 3, 3, padding=1, bias=False)
        self.deconv4_bn = nn.BatchNorm2d(3)
        self.subdeconv3 = nn.ConvTranspose2d(3, 3, 3, padding=1, bias=False)
        self.subdeconv3_bn = nn.BatchNorm2d(3)
        self.subdeconv4 = nn.ConvTranspose2d(3, 3, 3, padding=1, bias=False)
        self.subdeconv4_bn = nn.BatchNorm2d(3)

    def decode(self, x):
        z = self.z(x)
        z = z.view(-1, 512, 1, 1)

        deconv1 = self.lrelu(self.deconv1_bn(self.deconv1(self.up1(z))))
        deconv2 = self.up3(self.lrelu(self.deconv2_bn(self.deconv2(self.up2(deconv1)))))

        subdeconv1 = self.lrelu(self.subdeconv1_bn(self.subdeconv1(deconv2)))
        subdeconv2 = self.lrelu(self.subdeconv2_bn(self.subdeconv2(subdeconv1)))
        add1 = deconv2 + subdeconv2

        deconv3 = self.lrelu(self.deconv3_bn(self.deconv3(self.up4(add1))))
        deconv4 = self.deconv4_bn(self.deconv4(deconv3))

        subdeconv3 = self.lrelu(self.subdeconv3_bn(self.subdeconv3(deconv4)))
        subdeconv4 = self.lrelu(self.subdeconv4_bn(self.subdeconv4(subdeconv3)))

        add2 = deconv4 + subdeconv4
        decoded = self.tanh(add2)
        return (decoded+1)/2

    def forward(self, x):
        decoded = self.decode(x)
        return decoded
