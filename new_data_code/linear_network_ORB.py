from collections import OrderedDict
from torch import as_tensor, nn, no_grad, cat, flatten, float32, uint8, div, tensor
import torch
from torchvision import transforms
import numpy as np
import wandb
from random import sample

class BasicNetwork(nn.Module):
  def __init__(self, in_features, out_features):
    super(BasicNetwork,self).__init__()
    self.longer_stack = nn.Sequential(OrderedDict([ #Creating an ordered dictionary of the 3 layers we want in our NN
        ('Input', nn.Linear(in_features, 1024)),
        ('Relu 1', nn.ReLU()),
        ('Hidden Linear 1', nn.Linear(1024, 512)),
        ('Relu 2', nn.ReLU()),
        ('Hidden Linear 2', nn.Linear(512, 256)),
        ('Relu 3', nn.ReLU()),
        ('Hidden Linear 3', nn.Linear(256, 128)),
        ('Relu 4', nn.ReLU()),
        ('Hidden Linear 4', nn.Linear(128, 64)),
        ('Relu 5', nn.ReLU()),
        ('Hidden Linear 5', nn.Linear(64, 32)),
        ('Relu 6', nn.ReLU()),
        ('Hidden Linear 6', nn.Linear(32, 16)),
        ('Relu 7', nn.ReLU()),
        ('Hidden Linear 7', nn.Linear(16, 8)),
        ('Relu 8', nn.ReLU()),
        ('Output', nn.Linear(8, out_features))
    ]))
    self.wider_stack = nn.Sequential(OrderedDict([ #Creating an ordered dictionary of the 3 layers we want in our NN
        ('Input', nn.Linear(in_features, 8192)),
        ('Relu 1', nn.ReLU()),
        ('Hidden Linear 1', nn.Linear(8192, 4096)),
        ('Relu 2', nn.ReLU()),
        ('Hidden Linear 2', nn.Linear(4096, 1024)),
        ('Relu 3', nn.ReLU()),
        ('Hidden Linear 3', nn.Linear(1024, 16)),
        ('Relu 4', nn.ReLU()),
        ('Output', nn.Linear(8, out_features))
    ]))
    self.simple_stack = nn.Sequential(OrderedDict([ #Creating an ordered dictionary of the 3 layers we want in our NN
        ('Input', nn.Linear(in_features, 1024)),
        ('Relu 1', nn.ReLU()),
        ('Hidden Linear 1', nn.Linear(1024, 16)),
        ('Relu 2', nn.ReLU()),
        ('Output', nn.Linear(16, out_features))
    ]))

  #Defining how the data flows through the layer, PyTorch will call this we will never have to call it
  def forward(self, x):
    logits = self.longer_stack(x)
    return logits

# Model Evaluation #############################################################
#Takes in a dataloader, a NN model, our loss function, and an optimizer and trains the NN 
def train_loop(dataloader, model, loss_fn, optimizer, device, epoch, bs, will_save, key):
    batches = int(len(dataloader.dataset)*.8/bs)
    cumulative_loss = 0
    ret = []

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))
        cumulative_loss += loss
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if will_save and (batch % 10 == 0):
            range = sample(list(np.arange(len(X))), min(len(X), 20))
            for idx in range:
                new = torch.reshape(X[idx][:-3], (50,2))
                loc = X[idx][-3:]
                save = {"Train Key": key, "Sample Epoch":epoch,"Sample Training Loss":loss,
                "Diffs": new, 
                "Sample Training Latitude":loc[0], "Sample Training Longitude":loc[1], "Sample Training Altitude":loc[2], 
                "Sample Training Pred Lat": pred[idx][0].item(), "Sample Training Pred Lon": pred[idx][1].item(), "Sample Training Pred Alt": pred[idx][2].item(), 
                "Sample Training Truth Lat": y[idx][0].item(), "Sample Training Truth Lon": y[idx][1].item(), "Sample Training Truth Alt": y[idx][2].item()}
                ret.append(save)
                key+=1
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss} SAVED\n"
            print(row)
        else:
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss}\n"
            print(row)

    averages_1 = f"End of Testing \n Test Error: \n Testing Avg loss: {(cumulative_loss/batches):>8f}\n"
    print(averages_1)
    return ret, cumulative_loss/batches, key

#Runs through the whole dataset and gives final performace metrics
def test_loop(dataloader, model, loss_fn, device, epoch, bs, will_save, key):
    batches = int(len(dataloader.dataset)*.2/bs)
    cumulative_loss = 0
    ret = []

    with no_grad():
      for batch, (X, y) in enumerate(dataloader):
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))
        cumulative_loss += loss
        if will_save and (batch % 5 == 0):
            range = sample(list(np.arange(len(X))), min(len(X), 20))
            for idx in range:
                new = torch.reshape(X[idx][:-3], (50,2))
                loc = X[idx][-3:]
                save = {"Test Key": key, "Sample Epoch": epoch, "Sample Testing Loss":loss,
                "Diffs": new, 
                "Sample Testing Latitude":loc[0], "Sample Testing Longitude":loc[1], "Sample Testing Altitude":loc[2], 
                "Sample Testing Pred Lat": pred[idx][0].item(), "Sample Testing Pred Lon": pred[idx][1].item(), "Sample Testing Pred Alt": pred[idx][2].item(), 
                "Sample Testing Truth Lat": y[idx][0].item(),   "Sample Testing Truth Lon": y[idx][1].item(),   "Sample Testing Truth Alt": y[idx][2].item()}
                ret.append(save)
                key+=1
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss} SAVED\n"
            print(row) 
        else:
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss}\n"
            print(row)

    averages_1 = f"End of Testing \n Test Error: \n Testing Avg loss: {(cumulative_loss/batches):>8f}\n"
    print(averages_1)
    return ret, cumulative_loss/batches, key