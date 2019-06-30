import argparse
import numpy as np
import os
import scipy.io as sio
import torchvision.transforms as transforms

from glob import glob 
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets

import ernesgd_data.fid_score as fid_score


def download_real_images(img_size, batch_size, dataset_name, downloaded=False):
    
    data_dict = {'MNIST': datasets.MNIST,
            'CIFAR': datasets.CIFAR10}
    selected_dataset = data_dict[dataset_name]
    
    images_fname = dataset_name+'_images'
    real_images_fname = dataset_name+'_real_images'
    if not downloaded:
        os.makedirs(images_fname, exist_ok=True)
        os.makedirs(real_images_fname, exist_ok=True)

        # Download images
        dataloader = DataLoader(
            selected_dataset(images_fname,
                             train=True,
                             download=True,
                             transform=transforms.Compose(
                                 [transforms.Resize(img_size), transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])]
                             ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
    
        # Save real images
        for i, (imgs, _) in enumerate(dataloader):
            if i%10000 == 0:
                print(i)
            save_image(imgs, f"./{real_images_fname}/{i}.png", normalize=True)

    # Calculate stats (mu, Sigma) for generated images
    files = glob(real_images_fname+'/*.jpg') +glob(real_images_fname+'/*.png')
    FID_BATCH_SIZE = 200
    mu_real, Sigma_real = fid_score.calculate_activation_statistics(
            files, batch_size=FID_BATCH_SIZE, dims=2048, cuda=True, verbose=False)
    sio.savemat(f'{dataset_name}.mat', {'mu_real':mu_real, 'Sigma_real':Sigma_real})
    print('Done.')


if __name__ == "__main__":
    img_size = 32
    batch_size = 1
    dataset_name = 'CIFAR'
    downloaded = False
    download_real_images(img_size, batch_size, dataset_name, downloaded=downloaded)
