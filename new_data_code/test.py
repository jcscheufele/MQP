from sklearn.utils import shuffle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

model = torch.load("../../data/models/new/cv/model_Pair Normalized, single network, leaky, DS0 nonsbatch.pt", map_location=torch.device("cpu"))
dataset = torch.load("new_data/NYC_crossVal_test.pt", map_location=torch.device("cpu"))

with torch.no_grad():
    print(len(dataset))
    train_loader = DataLoader(dataset, batch_size=24998, shuffle=False)

    #for batch, (X, y) in enumerate(train_loader):
    batch, (X, y) = next(enumerate(train_loader))
    print(X)
    print(y)
    pred = model(X)

    mse = torch.mean(torch.square((pred-y[:, 2])))
    percent = torch.mean(torch.abs((pred-y[:,2])/y[:,2]))
    print(batch, percent.item(), mse.item())