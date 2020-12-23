import torch
import torchvision
from models.GAN import Generator, Discriminator
from torchsummary import summary

import os
import argparse

import numpy as np
import cv2

def run(p_seed=0, p_kernel_size=5, p_logdir="temp"):
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

    gen = Generator().to(device)

    gen.load_state_dict(torch.load(MODEL_FILE))
    gen.eval()

    sample = torch.randn((bs, n_gaussians)).float().cuda()
    gen_x = gen(sample)

    result = gen_x.clone().detach().cpu().numpy()
    result = np.reshape(result, (-1,28,28,1))*255
    for i in range(len(result)):
        cv2.imwrite('img/%d.png'%i, result[i])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--gpu", default=0, type=int)
    p.add_argument("--logdir", default="gan")
    args = p.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    run(p_seed = args.seed,
        p_logdir = args.logdir)

