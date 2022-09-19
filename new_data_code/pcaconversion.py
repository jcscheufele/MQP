import numpy as np
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from torch import as_tensor, div, flatten, cat
import torch
import cv2 as cv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from preprocessing import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock

from dataset_PCA import BasicDataset
from linear_network_PCA import BasicNetwork, train_loop, test_loop
#from conv_network import BasicNetwork, train_loop, test_loop
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

dataset = torch.load("new_data/samepic.pt")
print("data loaded")

pcatrans= torch.pca_lowrank(torch.as_tensor(dataset.X))

print(pcatrans)

"""np_arr = np.asarray(dataset.X)

print(np_arr)

sc = StandardScaler()
pca = PCA(n_components=5)
np_arr = sc.fit_transform(np_arr)
np_arr = pca.fit_transform(np_arr)"""

