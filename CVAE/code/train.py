import torch
import torch.nn.functional as F

import torchvision
from torchvision import datasets
from torchvision import transforms

from models.CVAE_dense import CVAE
import numpy as np
import cv2
from torchsummary import summary
import argparse
import os

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())

    return BCE, KLD

def run(p_seed=0, p_epochs=150, p_logdir="temp"):
    # random number generator seed ------------------------------------------------#
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

    if not os.path.exists("imgs"):
        os.makedirs("imgs")

    bs = 128
    epochs = p_epochs

    # Load Data
    dataset = datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

    n_gaussians = 100
    vae = CVAE(n_gaussians).to(device)

    #summary(vae, (1,28,28))
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device).view(-1, 28*28), target.to(device)
            recon_images, mu, logvar = vae(data, target)
            bce, kld = loss_fn(recon_images, data, mu, logvar)
            loss = bce + kld
            loss = loss / data.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if idx%100 == 0:
                print("Epoch[{}/{}] Loss: {:.3f} {:.3f}".format(epoch+1, epochs, bce.item()/data.size(0), kld.item()/data.size(0)))

        # Save results -------------------------------------------------------------#
        result = recon_images.clone().detach().cpu().numpy()
        result = np.reshape(result, (-1, 28,28,1))*255
        for i in range(len(result)):
            cv2.imwrite('img/%d.png'%i, result[i])
        
        # Save Model parameter -----------------------------------------------------#
        torch.save(vae.state_dict(), MODEL_FILE)
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--epochs", default=30, type=int)    
    p.add_argument("--gpu", default=0, type=int)
    p.add_argument("--logdir", default="cvae")
    args = p.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    run(p_seed = args.seed,
        p_epochs = args.epochs,
        p_logdir = args.logdir)
