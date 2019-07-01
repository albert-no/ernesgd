import torch.nn as nn
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, img_size, channels, latent_dim):
        super(Generator, self).__init__()
        #let N = img_size (must be divisible by 4)
        self.init_size = img_size // 4
        
        #latent_dim to 128 * (N / 4)^2 fully-connected layer
        #Why is layer 1 written separately?
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        #XXX DCGAN paper uses non-leaky ReLu layers for generator
        #XXX DCGAN paper uses "project and reshape" for first layer. Is this the same?
        #XXX DCGAN paper describe the convolution operation as "fractinally-strided" convolutions
        #XXX which would corredpond to conv2d_transpose(...) in PyTorch

        #after reshape,  128x(N/4)x(N/4)
        #after upsample, 128x(N/2)x(N/2)
        #after conv2d,   128x(N/2)x(N/2)
        #after upsample, 128xNxN
        #after conv2d,   64xNxN
        #after conv2d,   channelsxNxN
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
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)

        
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            #kernelsize=3, stride=2, padding=1 (reduces image size by half)
            #XXX DCGAN paper does not use Dropout
            #XXX use of inplace is discouraged
            #https://pytorch.org/docs/master/notes/autograd.html#in-place-operations-on-variables
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        #input channelxNxN
        #after conv2d 16x(N/2)x(N/2)
        #after conv2d 32x(N/4)x(N/4)
        #after conv2d 64x(N/8)x(N/8)
        #after conv2d 128x(N/16)x(N/16)
        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // (2 ** 4)

        #fully connected layer from 128(N/16)(N/16) to 1 composed with sigmoid
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        
        #reshape  128x(N/16)x(N/16) to 128(N/16)(N/16)
        out = out.view(out.shape[0], -1)
        
        validity = self.adv_layer(out)

        return validity

