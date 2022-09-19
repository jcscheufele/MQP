import numpy as np
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from torch import as_tensor, div, flatten, cat
import torch
import cv2 as cv
import pandas as pd

from preprocessing import *

from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock

# search is the string you're looking for inside the log file and path is the path to the log file on your computer.
# This function pulls out the key information from each line with they key, search string.  In our case it pulls out the altitude, latitude, and longtitude of each image location.

def createTransform():
    p144 = (144, 256)
    p240 = (240, 426)
    p360 = (360, 640)
    p480 = (480, 848)
    crop = (355, 644)
    listOfTransforms = [
        transforms.Resize(p240)
        ]
    return transforms.Compose(listOfTransforms)

class BasicDataset(Dataset):
    def __init__(self, data_dir):
        self.data = pd.read_csv(data_dir, index_col=0)
        self.transform = createTransform()
        self.X = []
        #self.makeNewFeature()
        self.__makefeatures__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], as_tensor(self.data[['lon_dif', 'lat_dif', 'alt_dif']].iloc[idx], dtype=torch.float32)

    def __getimage__(self, path):
        return read_image(path, mode=ImageReadMode.GRAY)

    def __makefeatures__(self):
        print("Making Features...")
        for idx in range(self.__len__()-2):
            print(f"[ {idx}/{self.__len__()} ]", end='\r')

            img1 = self.__getimage__(self.data.images.iloc[idx])
            img2 = self.__getimage__(self.data.images_new.iloc[idx])
            loc = [self.data.lat.iloc[idx], self.data.lon.iloc[idx], self.data.alt.iloc[idx]]

            img1 = self.transform(img1)
            img2 = self.transform(img2)

            img3 = cat((img1,img2), 1)
            #print("img3 shape", img3.shape)
            
            #cv.imwrite("test_img1.png", img1[0].numpy())
            #cv.imwrite("test_img2.png", img2[0].numpy())
            #cv.imwrite("test_img3.png", img3[0].numpy())

            eq_img = cv.equalizeHist(img3[0].numpy())
            #cv.imwrite("test_eq_img.png", eq_img)

            #img1 = clahe_method(img1[0].numpy())
            #img2 = clahe_method(img2[0].numpy())

            eq_img = as_tensor(eq_img, dtype=torch.uint8)

            eq_img = div(eq_img, 255)

            eq_img = flatten(eq_img)
            #img2 = flatten(img2)

            data_input = cat((eq_img, as_tensor(loc, dtype=torch.float32)))
            #print("Input:", data_input.shape)
            
            self.X.append(data_input)
            #break
            #exx.append(torch.unsqueeze(input, dim=0)) #flattens each image and concatonates them and appends them to a list
        print("Features Created")