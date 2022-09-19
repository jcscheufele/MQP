from dataset_unnormalized import BasicDataset
from linear_network_alt import BasicNetwork, train_loop, test_loop
#from conv_network import BasicNetwork, train_loop, test_loop
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import torch
import wandb

MAINPATH = '/work/shared/DEVCOM-SC21/Network'
datatype = 'same_pic'
save_dir_data = MAINPATH + f'/data/datasets/{datatype}/'
img_dir = MAINPATH + f'/data/images/{datatype}/HighresScreenshot'
log_dir = MAINPATH + f'/data/logs/{datatype}/SplinePath.log'

#np.set_printoptions(threshold=sys.maxsize)
if __name__ == "__main__":
    wandb.init(project="test-project", entity="parachuteproject2021")
    new_name = "UN, greaterthan0, 75000 DS2 200m, with percent, alt only"
    wandb.run.name = new_name
    wandb.run.save()

    #dataset = BasicDataset("../../data/training_75000pix_200m_greaterthan0_1.csv")
    #torch.save(dataset, "new_data/training_75000_pairNorm__greaterthan0_1.pt")
    #print("data saved 1")training_75000_pairNorm__greaterthan0_0.pt

    dataset = torch.load("new_data/training_75000_UN__greaterthan0_2.pt")
    print("data loaded")

    shuffle = True
    validation_split = .2
    batch_size = 128
    epochs = 750
    learningrate = 0.001

    wandb.config = {
    "learning_rate": learningrate,
    "epochs": epochs,
    "batch_size": batch_size
    }

    dataset_size = len(dataset)-2
    indices = list(range(dataset_size)) # creates a list that creates a list of indices of the dataset
    split = int(np.floor(validation_split * dataset_size)) #Finding the index to split the dataset at given percentent inputted above
    
    if shuffle: #if shuffle is chosen, it will shuffle before it is split
        np.random.seed(112) #sets how it will be shuffles
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]  #splits the dataset and assigns them to training and testing datasets

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices) 
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    print(len(train_loader.dataset))
    print(len(valid_loader.dataset))

    in_features = (240*426*2)+3
    out_features = 1
    print(in_features, out_features)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = BasicNetwork(in_features, out_features).to(device)
    print(model)

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

    tr_key = 0
    te_key = 0
    will_save = False
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        if (epoch % int(epochs/10)) == 0:
            will_save = True
        else:
            will_save = False

        training_dicts, train_error, train_percent, tr_key = train_loop(train_loader, model, loss_fn, optimizer, device, epoch, batch_size, will_save, tr_key)
        testing_dicts, test_error, test_percent, te_key = test_loop(valid_loader, model, loss_fn, device, epoch, batch_size, will_save, te_key)
        if will_save:
            for dict1, dict2 in zip(training_dicts, testing_dicts):
                wandb.log(dict1)
                wandb.log(dict2)
        wandb.log({"Epoch Training Loss":train_error, "Epoch Testing Loss": test_error,
        "Epoch Training Alt Percent": train_percent, "Epoch Testing Alt Percent": test_percent, 
        "Epoch epoch":epoch})

    save_loc = f"../../data/models/new/model_{new_name}.pt"
    print(f"saving Network to {save_loc}")
    torch.save(model, save_loc)