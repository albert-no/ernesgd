#!python
#!/usr/bin/env python
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk("."):
    for file in f:
        if '.mat' in file:
            loaded_file = loadmat(file)
            #print(loaded_file)
            score = loaded_file['FID'][0]
            plt.plot(range(0,len(score)),score,'.-')
            plt.xlabel('Epochs')
            plt.ylabel('FID')
            plt.savefig("{}.png".format(file[0:-4]))
            plt.close()


            
