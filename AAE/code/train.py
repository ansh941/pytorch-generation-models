import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets
from torchvision import transforms

from models.AAE import Decoder, Discriminator, Encoder
import numpy as np
import cv2
from torchsummary import summary
import argparse
import os

def run(p_seed=0, p_epochs=150, p_logdir="temp"):
    # random number decerator seed ------------------------------------------------#
    SEED = p_seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # enable GPU usage ------------------------------------------------------------#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")
        exit(0)

    # file names ------------------------------------------------------------------#
    if not os.path.exists("../logs/%s"%p_logdir):
        os.makedirs("../logs/%s"%p_logdir)
    MODEL_FILE = str("../logs/%s/model%03d.pth"%(p_logdir,SEED))

    if not os.path.exists("img"):
        os.makedirs("img")

    bs = 128
    epochs = p_epochs

    # Load Data
    dataset = datasets.MNIST(root='../data/mnist', train=True, transform=transforms.ToTensor(), download=True)
    #dataset = datasets.CIFAR10(root='../data/cifar10', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

    dis = Discriminator().to(device)
    dec = Decoder().to(device)
    enc = Encoder().to(device)

    #summary(dis, (1,100))
    #summary(dec, (1,100))
    #summary(enc, (1,28*28))

    dis_optimizer = torch.optim.Adam(dis.parameters(), lr=1e-3)
    dec_optimizer = torch.optim.Adam(dec.parameters(), lr=1e-4)
    enc_optimizer = torch.optim.Adam(enc.parameters(), lr=1e-4)
    loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for idx, (data, _) in enumerate(data_loader):
            dis_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            data = data.to(device)
            fake_label = torch.zeros(data.size(0)).to(device).long()
            real_label = torch.ones(data.size(0)).to(device).long()

            real_z = torch.randn((data.size(0), 100)).to(device)

            # AutoEncoder Training
            fake_z = enc(data)
            recon_x = dec(fake_z)

            recon_loss = F.binary_cross_entropy(recon_x, data)

            recon_loss.backward(retain_graph=True)
            enc_optimizer.step()
            dec_optimizer.step()

            # Discriminator Training
            dis_real = dis(real_z)
            dis_fake = dis(fake_z.detach())

            d_real_loss = loss(dis_real, real_label)
            d_fake_loss = loss(dis_fake, fake_label)

            d_loss = (d_real_loss + d_fake_loss)/2

            d_loss.backward()

            dis_optimizer.step()
    
            # Generator training
            fake_z = enc(data)
            dis_fake = dis(fake_z.detach())
            g_loss = loss(dis_fake, real_label)
            g_loss.backward(retain_graph=True)

            dec_optimizer.step()
               
            if idx%100 == 0:
                print("Epoch[{}/{}] Loss: {:.3f} {:.3f}".format(epoch+1, epochs, recon_loss, d_loss))

        # Save results -------------------------------------------------------------#
        result = recon_x.clone().detach().cpu().numpy()
        #result = np.reshape(result, (-1,28,28,1))*255
        result = np.transpose(result, [0,2,3,1])*255
        for i in range(len(result)):
            cv2.imwrite('img/%d.png'%i, result[i])
        
        # Save Model parameter -----------------------------------------------------#
        torch.save(dec.state_dict(), MODEL_FILE)
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--epochs", default=30, type=int)    
    p.add_argument("--gpu", default=0, type=int)
    p.add_argument("--logdir", default="aae")
    args = p.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    run(p_seed = args.seed,
        p_epochs = args.epochs,
        p_logdir = args.logdir)
