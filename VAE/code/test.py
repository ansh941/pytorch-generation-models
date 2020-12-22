import torch
import torchvision
from models.VAE import VAE
from torchsummary import summary

import os
import argparse

import numpy as np
import cv2

def run(p_seed=0, p_epochs=150, p_kernel_size=5, p_logdir="temp"):
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
    MODEL_FILE = str("../logs/%s/model%03d.pth"%(p_logdir,SEED))

    bs = 10

    n_gaussians = 100

    vae = VAE(n_gaussians).to(device)

    vae.load_state_dict(torch.load(MODEL_FILE))
    vae.eval()

    sample = torch.randn((bs, n_gaussians)).float().cuda()
    recon_x = vae.decoder(sample)

    result = recon_x.clone().detach().cpu().numpy()
    result = np.transpose(result, [0,2,3,1])*255
    for i in range(len(result)):
        cv2.imwrite('img/%d.png'%i, result[i])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--epochs", default=150, type=int)
    p.add_argument("--gpu", default=0, type=int)
    p.add_argument("--logdir", default="vae")
    args = p.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    run(p_seed = args.seed,
        p_epochs = args.epochs,
        p_logdir = args.logdir)

