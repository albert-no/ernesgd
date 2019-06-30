import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import scipy.io as sio

import pathlib

import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_images", type=int, default=10000, help="number of images to generate")
parser.add_argument("--batch_size", type=int, default=100, help="number of images to gnerate per iteration")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_ckpt", type=int, default=200, help="dimensionality of the latent space")
opt = parser.parse_args()
print(opt)

# parser = argparse.ArgumentParser()
# parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
# opt = parser.parse_args()
# print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

PATH = './checkpoints/ckpt{}.pt'.format(str(opt.n_ckpt))
generator = Generator()
generator.load_state_dict(torch.load(PATH))

# generator = torch.load('./checkpoints/ckpt.pt')
# generator.eval()

if cuda:
    generator.cuda()

# FID evaluation parameters
FID_EVAL_SIZE = opt.n_images # Number of samples for evaluation
FID_SAMPLE_BATCH_SIZE = opt.batch_size  # Batch size of generating samples, lower to save GPU memory

samples = Tensor(np.zeros((FID_EVAL_SIZE, opt.channels, opt.img_size, opt.img_size)))
n_fid_batches = FID_EVAL_SIZE // FID_SAMPLE_BATCH_SIZE

# generate fake images
for i in range(n_fid_batches):

    print("\rgenerate fid sample batch %d/%d " % (i + 1, n_fid_batches), end="", flush=True)

    frm = i * FID_SAMPLE_BATCH_SIZE
    to = frm + FID_SAMPLE_BATCH_SIZE

    # Sample noise as generator input
    z = Variable(Tensor(np.random.normal(0, 1, (FID_SAMPLE_BATCH_SIZE, opt.latent_dim))))

    imgs = generator(z)
    print(imgs.shape)

    # Generate a batch of images
    samples[frm:to] = generator(z)

print(type(samples))
print(samples.shape)


# # generate fake images
# for i in range(n_fid_batches):

#     print("\rgenerate fid sample batch %d/%d " % (i + 1, n_fid_batches), end="", flush=True)

#     frm = i * FID_SAMPLE_BATCH_SIZE
#     to = frm + FID_SAMPLE_BATCH_SIZE

#     # Sample noise as generator input
#     z = Variable(Tensor(np.random.normal(0, 1, (FID_SAMPLE_BATCH_SIZE, opt.latent_dim))))

#     # Generate a batch of images
#     gen_imgs = generator(z)

#     # save all generated images to the file folder ''test_gen_temp_images''
#     # save_image(gen_imgs.data[:1], "test_images_one_digit/%d.png" % i, normalize=True)
#     save_image(gen_imgs.data, "./test_images/%d.png" % i, nrow=5, normalize=True)
