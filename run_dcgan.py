import argparse
import os
import numpy as np
import math
import scipy.io as sio
import pathlib

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim 

import torch.nn as nn
import torch.nn.functional as F
import torch

from ernesgd_custom.ernesgd import ERNESGD
from ernesgd_utils.dcgan import weights_init_normal, Generator, Discriminator
import ernesgd_data.fid_score as fid_score


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--optimizer", type=str, default='ernesgd',
        help="optimizer to use, (ernesgd or sgd)")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--no_fid_score", dest='no_fid_score', action='store_true',
        help="skip computing fid_score")

# XXX additionals for FID
parser.add_argument("--FID_epochs", type=int, default=1,
        help="evaluate the FID score for every FID_epochs epochs")
parser.add_argument("--n_epochs_interval_save_ckpt", type=int, default=10,
        help="epoch interval to save checkpoints")
parser.add_argument("--dataset_name", type=str, default='MNIST',
        help="name of data (MNIST or CIFAR)")
parser.set_defaults(no_fid_score=False)
opt = parser.parse_args()


if opt.dataset_name == 'CIFAR':
    opt.channels = 3
    opt.img_size = 32  #we need 64?
    opt.latent_dim = 100
if opt.dataset_name == 'MNIST':
    opt.channels = 1
    opt.img_size = 32   #img_size needs to be divisible by 16.
                        #However, MNIST image is 28x28. Do we do zero padding? XXX
    opt.latent_dim = 100
    
print(opt)


# XXX Generate output folders
folder_names = ['images', 'chk', 'fid', 'temp']
out_path = os.path.join(*[os.getcwd(), 'outputs',
    f'{opt.optimizer}_lr{opt.lr}_data{opt.dataset_name}'])
out_fnames = {}
for folder_name in folder_names:
    out_fnames[folder_name] = os.path.join(out_path, folder_name)
    os.makedirs(out_fnames[folder_name], exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(opt.img_size, opt.channels, opt.latent_dim)
discriminator = Discriminator(opt.img_size, opt.channels)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
data_folder_name = f'ernesgd_data/{opt.dataset_name}/'
os.makedirs(data_folder_name, exist_ok=True)
data_dict = {'MNIST': datasets.MNIST,
        'CIFAR': datasets.CIFAR10}

selected_dataset = data_dict[opt.dataset_name]
#transforms.Resize changes the image dimension
dataloader = torch.utils.data.DataLoader(
    selected_dataset(data_folder_name,
                     train=True,
                     download=True,
                     transform=transforms.Compose(
                         [transforms.Resize(opt.img_size), transforms.ToTensor(),
                             transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers XXX use ERNESGD
optimizer_dict = {'ernesgd': ERNESGD, 'sgd': optim.SGD}
selected_optimizer = optimizer_dict[opt.optimizer]
optimizer_G = selected_optimizer(generator.parameters(), lr=opt.lr)
optimizer_D = selected_optimizer(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
FID_EVAL_SIZE = 10000  # Number of images generated for FID score evaluation
FID_SAMPLE_BATCH_SIZE = 1  # Batch size of generating samples, lower to save GPU memory
FID_BATCH_SIZE = 200  # Batch size fo final FID calculation, i.e., inception propagation etc.

real_stat= sio.loadmat(f'{opt.dataset_name}.mat')
mu_real, Sigma_real = real_stat['mu_real'], real_stat['Sigma_real']
FID_record = []

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        if opt.optimizer == 'ernesgd':
            # XXX First half step
            optimizer_G.half_step()

            # XXX Second half step
            optimizer_G.zero_grad()

            # XXX is it necessary to do this again?
            # Generate a batch of images XXX using same z with first half step
            gen_imgs = generator(z)
        
            # XXX is it necessary to do this again?
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.full_step()
        elif opt.optimizer == 'sgd':
            optimizer_G.step()
        else:
            raise ValueError("optimizer should be either 'ernesgd' or 'sgd'")
        

        # ---------------------
        #  Train Discriminator
        # ---------------------


        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        if opt.optimizer == 'ernesgd':
            # XXX First half step
            optimizer_D.half_step()

            # XXX Second half step
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.full_step()
        elif opt.optimizer == 'sgd':
            optimizer_D.step()
        else:
            raise ValueError("optimizer should be either 'ernesgd' or 'sgd'")


        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], out_fnames['images']+f"/{batches_done}.png", nrow=5, normalize=True)

    if (not opt.no_fid_score) and epoch % opt.n_epochs_interval_save_ckpt == 0:
        # torch.save(generator, './checkpoints/ckpt.pt')
        PATH = f"{out_fnames['chk']}/ckpt{epoch}.pt"
        torch.save(generator.state_dict(), PATH)

    # ------------------------------------
    # We evaluate FID score for each epoch
    # ------------------------------------

    if (not opt.no_fid_score) and epoch % opt.FID_epochs == 0:
        n_fid_batches = FID_EVAL_SIZE // FID_SAMPLE_BATCH_SIZE

        for i in range(n_fid_batches):

            print(f"\rgenerate fid sample batch {i+1}/{n_fid_batches} ", end="", flush=True)

            # frm = i * FID_SAMPLE_BATCH_SIZE
            # to = frm + FID_SAMPLE_BATCH_SIZE

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (FID_SAMPLE_BATCH_SIZE, opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # save all generated images to the file folder ''temp_images''
            # this step is not essentially necessary since we don't have to save images first and then
            # calculate the FID score. But at this moment I save images first to aovid changing too much codes 
            # in the given FID score function.
            save_image(gen_imgs.data, f"{out_fnames['temp']}/{i}.png", normalize=True)

        path = pathlib.Path(out_fnames['temp'])
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

        # calculate stats (mu, Signa) for the generated pictures
        print('evaluating the FID score now ...')
        mu_gen, Sigma_gen = fid_score.calculate_activation_statistics(
                files, batch_size=FID_BATCH_SIZE, dims=2048, cuda=True, verbose=False)
        mu_real = mu_real.reshape(-1)

        # calculate the FID score
        FID = fid_score.calculate_frechet_distance(
            mu_real, Sigma_real, mu_gen, Sigma_gen, eps=1e-6)
        FID_record.append(FID)
        print('\n The FID score collected until now is', FID_record)
        sio.savemat(out_fnames['fid']+'/fid.mat', {'FID':FID_record}) # used to plot the curve

os.system(f"rm -r {out_fnames['temp']}")
