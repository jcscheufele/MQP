from collections import OrderedDict
from torch import as_tensor, nn, no_grad, cat, flatten, float32, uint8, div, tensor, mean, reshape, unsqueeze
from torchvision import transforms
import numpy as np
import wandb
from random import sample


class BasicNetwork(nn.Module):
  def __init__(self, in_features, out_features):
    super(BasicNetwork,self).__init__()
    self.convolution = nn.Sequential(OrderedDict([
    ("Input", nn.Conv2d(2, 8, kernel_size=10, padding=1)),
    ("ReLU 1", nn.ReLU()),
    ("Max Pooling 1", nn.MaxPool2d(4, 4)),

    ("Hidden 1", nn.Conv2d(8, 32, kernel_size=10, padding=1)),
    ("ReLU 2", nn.ReLU()),
    ("Max Pooling 2", nn.MaxPool2d(2, 2)),

    ("Hidden 2", nn.Conv2d(32, 128, kernel_size=3, padding=1)),
    ("ReLU 3", nn.ReLU()),
    ("Max Pooling 3", nn.MaxPool2d(2, 2)),

    ("Hidden 3", nn.Conv2d(128, 128, kernel_size=3, padding=1)),
    ("ReLU 4", nn.ReLU()),
    ("Max Pooling 4", nn.MaxPool2d(2, 2)),
    ("Output Flatten", nn.Flatten())
    ]))
    self.longer_stack = nn.Sequential(OrderedDict([ #Creating an ordered dictionary of the 3 layers we want in our NN
        ('Input', nn.Linear(9219, 1024)),
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

  #Defining how the data flows through the layer, PyTorch will call this we will never have to call it
  def forward(self, x):   # (62, 128, 440*426+3) - > (62, 128, 240, 426), (62, 128, 240, 426)
    #print(x.size())
    split = int((x.size()[1]-3)/2)
    #print(split)
    img1 = unsqueeze(reshape(x[:, :split], (x.size()[0], 240, 426)), dim=1)
    img2 = unsqueeze(reshape(x[:, split:-3], (x.size()[0], 240, 426)), dim=1)
    #print(img1.size(), img2.size())
    loc = reshape(x[:, -3:], (x.size()[0], -1))
    inp = cat((img1, img2), 1)
    #print(inp.size())
    #print(loc.size())
    logits = self.convolution(inp)
    #print(logits.size())
    logits = cat((logits, loc), 1)
    #print(logits.size())
    logits = self.longer_stack(logits)
    return logits

# Model Evaluation #############################################################
#Takes in a dataloader, a NN model, our loss function, and an optimizer and trains the NN 
def train_loop(dataloader, model, loss_fn, print_loss_fn, optimizer, device, epoch, bs, will_save, key):
    batches = int(len(dataloader.dataset)*.8/bs)
    cumulative_loss = 0
    cum_alt_loss, cum_lat_loss, cum_lon_loss = 0,0,0
    cum_alt_percent, cum_lat_percent, cum_lon_percent = 0,0,0
    ret = []

    for batch, (X, y) in enumerate(dataloader):
        y=y.to(device)
        pred = model(X.to(device))
        loss = loss_fn(pred, y)
        print_loss = print_loss_fn(pred, y)

        lat_percent = abs(div(pred[:, 0]-y[:, 0], y[:,0]))
        lon_percent = abs(div(pred[:, 1]-y[:, 1], y[:,1]))
        alt_percent = abs(div(pred[:, 2]-y[:, 2], y[:,2]))

        cumulative_loss += loss
        cum_lat_loss += mean(print_loss[:,0]).item()
        cum_lon_loss += mean(print_loss[:,1]).item()
        cum_alt_loss += mean(print_loss[:,2]).item()

        cum_lat_percent += mean(lat_percent).item()
        cum_lon_percent += mean(lon_percent).item()
        cum_alt_percent += mean(alt_percent).item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if will_save and (batch % 10 == 0):
            range = sample(list(np.arange(len(X))), min(len(X), 20))
            for idx in range:
                split = int((len(X[idx])-3)/2)
                img1 = X[idx][:split].numpy().reshape((240,426))
                img2 = X[idx][split:-3].numpy().reshape((240,426))
                loc = X[idx][-3:]

                save = {"Train Key": key, "Sample Epoch":epoch,"Sample Training Loss":loss,
                "Sample Training First Image": wandb.Image(img1), "Sample Training Second Image": wandb.Image(img2), 
                "Sample Training Latitude":loc[0], "Sample Training Longitude":loc[1], "Sample Training Altitude":loc[2], 
                "Sample Training Pred Lat": pred[idx][0].item(), "Sample Training Pred Lon": pred[idx][1].item(), "Sample Training Pred Alt": pred[idx][2].item(), 
                "Sample Training Truth Lat": y[idx][0].item(), "Sample Training Truth Lon": y[idx][1].item(), "Sample Training Truth Alt": y[idx][2].item(),
                "Sample Training Percent Err Lat": lat_percent[idx].item(), "Sample Training Percent Err Lon": lon_percent[idx].item(), "Sample Training Percent Err Alt": alt_percent[idx].item()}
                ret.append(save)
                key+=1
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss} SAVED\n"
            print(row) 
        else:
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss}\n"
            print(row)

    averages_1 = f"End of Testing \n Test Error: \n Testing Avg loss: {(cumulative_loss/batches):>8f}\n"
    print(averages_1)
    return ret, cumulative_loss/batches, cum_lat_loss/batches, cum_lon_loss/batches, cum_alt_loss/batches, cum_lat_percent/batches, cum_lon_percent/batches, cum_alt_percent/batches, key

#Runs through the whole dataset and gives final performace metrics
def test_loop(dataloader, model, loss_fn, print_loss_fn, device, epoch, bs, will_save, key):
    batches = int(len(dataloader.dataset)*.2/bs)
    cumulative_loss = 0
    cum_alt_loss, cum_lat_loss, cum_lon_loss = 0,0,0
    cum_alt_percent, cum_lat_percent, cum_lon_percent = 0,0,0
    ret = []

    with no_grad():
      for batch, (X, y) in enumerate(dataloader):
        pred = model(X.to(device))
        y = y.to(device)
        loss = loss_fn(pred, y)
        print_loss = print_loss_fn(pred, y)            

        lat_percent = abs(div(pred[:, 0]-y[:, 0], y[:,0]))
        lon_percent = abs(div(pred[:, 1]-y[:, 1], y[:,1]))
        alt_percent = abs(div(pred[:, 2]-y[:, 2], y[:,2]))

        cumulative_loss += loss
        cum_lat_loss += mean(print_loss[:,0]).item()
        cum_lon_loss += mean(print_loss[:,1]).item()
        cum_alt_loss += mean(print_loss[:,2]).item()

        cum_lat_percent += mean(lat_percent).item()
        cum_lon_percent += mean(lon_percent).item()
        cum_alt_percent += mean(alt_percent).item()

        if will_save and (batch % 5 == 0):
            range = sample(list(np.arange(len(X))), min(len(X), 20))
            for idx in range:
                split = int((len(X[idx])-3)/2)
                img1 = X[idx][:split].numpy().reshape((240,426))
                img2 = X[idx][split:-3].numpy().reshape((240,426))
                loc = X[idx][-3:]

                save = {"Test Key": key, "Sample Epoch": epoch, "Sample Testing Loss":loss,
                "Sample Testing First Image": wandb.Image(img1), "Sample Testing Second Image": wandb.Image(img2), 
                "Sample Testing Latitude":loc[0], "Sample Testing Longitude":loc[1], "Sample Testing Altitude":loc[2], 
                "Sample Testing Pred Lat": pred[idx][0].item(), "Sample Testing Pred Lon": pred[idx][1].item(), "Sample Testing Pred Alt": pred[idx][2].item(), 
                "Sample Testing Truth Lat": y[idx][0].item(),   "Sample Testing Truth Lon": y[idx][1].item(),   "Sample Testing Truth Alt": y[idx][2].item(),
                "Sample Testing Percent Err Lat": lat_percent[idx].item(), "Sample Testing Percent Err Lon": lon_percent[idx].item(), "Sample Testing Percent Err Alt": alt_percent[idx].item()}
                ret.append(save)
                key+=1
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss} SAVED\n"
            print(row) 
        else:
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss}\n"
            print(row)

    averages_1 = f"End of Testing \n Test Error: \n Testing Avg loss: {(cumulative_loss/batches):>8f}\n"
    print(averages_1)
    return ret, cumulative_loss/batches, cum_lat_loss/batches, cum_lon_loss/batches, cum_alt_loss/batches, cum_lat_percent/batches, cum_lon_percent/batches, cum_alt_percent/batches, key