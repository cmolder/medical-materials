import os, sys

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

import mmac_net
from datasets import PatchNpyDataset
from mmac_net.train_helpers import loss_acc



BATCH_SIZE  = 50   # Number of pos/neg pairs per batch
LR_START    = 1e-4 # Initial learning rate
LR_PATIENCE = 1    # Number of epochs of training loss increase before the learning rate is reduced by 10x
EPOCHS      = 15   # Number of epochs

# The type of MMAC_CNN model to be trained
# - mmac_net.MMAC_CNN       : ResNet34
# - mmac_net.MMAC_CNN_VGG16 : VGG16
MMAC_MODEL  = mmac_net.MMAC_CNN_VGG16

def train(model, device, train_loader, optimizer, 
          w_per = 1e0, w_kld = 1e-2, verbose=True):
    """Trains the MMAC_MODEL one time.

    Parameters:
        model: MMAC_MODEL
            The neural network being tested.
        device: string
            The device (cpu or cuda) that the model will be run on.
        train_loader: DataLoader
            The training data.
        optimizer: torch.optim
            The optimizer used to modify the weights of the model.
        w_kld: int
            KL-divergence hyperparameter
            w_a in SchNis (8)
        w_per: int
            w_2 in SchNis (8)
        verbose: bool (optional)
            If True, prints out information (in addition to progress 
            bars) from the function.
            
    Returns:
        total_loss: float
            The total loss from training the MMAC-CNN.
        accuracy: float
            The accuracy of the MMAC-CNN's predictions during training.
    """
    model.train()
    
    A = model.A  
    total_loss  = 0.0
    total_acc   = 0.0
    total_ucost = 0.0

    for batch_idx, batch in enumerate(tqdm(train_loader, unit=' training batches')):
        X, y = batch
        
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        # y_pred is the k class prediction
        # A_preds are the a1, a2, ..., a5, a_final m attribute predictions
        # from each level of the auxillary layers
        y_pred, A_preds = model(X)
        
        # print(f'{y_pred.shape} --> k preds')
        # print(f'{y.shape} --> k labels')
        # for pred_idx, pred in enumerate(A_preds):
        #     print(f'{pred.shape} --> m preds {pred_idx}')
        
        # Loss accuracy evaluation
        # l is the loss
        # a is the accuracy
        # u is the u-cost
        # y is a k-dimensional one-hot vector
        # A is the (k x m) matrix      
        loss, acc, u_cost = loss_acc(y, y_pred, A, A_preds, w_kld, w_per)
        
        # Backwards pass on the loss
        loss.backward()
        optimizer.step()
        
        
        # Set a new learning rate
        # set_lr(optim, cosine_lr(count + batch_idx, lr_cycles * count, min_lr, max_lr))
        total_loss  += float(loss)
        total_acc   += acc
        total_ucost += u_cost
        
    # Print results
    if verbose:
        print(f'\nTrain loss  : {total_loss / len(train_loader)}')
        print(f'Train ucost : {total_ucost / len(train_loader)}')
        print(f'Train acc   : {total_acc / len(train_loader) * 100:.3f}%')
    
    return total_loss, total_acc / len(train_loader), total_ucost / len(train_loader)
    
    
    
def validate(model, device, val_loader,
             w_per = 1e0, w_kld = 1e-2, verbose=True):
    """Runs the MMAC_MODEL one time over the validation set.

    Parameters:
        model: MMAC_MODEL
            The neural network being tested.
        device: string
            The device (cpu or cuda) that the model will be run on.
        val_loader: DataLoader
            The validation data.
        w_kld: int
            KL-divergence hyperparameter
            w_a in SchNis (8)
        w_per: int
            w_2 in SchNis (8)
        verbose: bool (optional)
            If True, prints out information (in addition to progress 
            bars) from the function.
            
    Returns:
        total_loss: float
            The total loss from evaluating the MMAC-CNN on the validation set.
        accuracy: float
            The accuracy from evaluating the MMAC-CNN on the validation set.
    """
    model.eval()
    
    A = model.A
    total_loss  = 0.0
    total_acc   = 0.0
    total_ucost = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, unit=' validation batches')):
            X, y = batch
    
            X = X.to(device)
            y = y.to(device)
            
            # Forward pass
            # y_pred is the k class prediction
            # A_preds are the a1, a2, ..., a5, a_final m attribute predictions
            # from each level of the auxillary layers
            y_pred, A_preds = model(X)
            
            # l is the loss
            # a is the accuracy
            # u is the u-cost
            # y is a k-dimensional one-hot vector
            # A is the (k x m) matrix      
            loss, acc, u_cost = loss_acc(y, y_pred, A, A_preds, w_kld, w_per)
            
            total_loss  += float(loss)
            total_acc   += acc
            total_ucost += u_cost

    # Print results
    if verbose:
        print(f'\nVal loss  : {total_loss / len(val_loader)}')
        print(f'Val ucost : {total_ucost / len(val_loader)}')
        print(f'Val acc   : {total_acc / len(val_loader) * 100:.3f}%')
    
    return total_loss, total_acc / len(val_loader), total_ucost / len(val_loader)
        


def main(data_root):
    """Driver class for training the MMAC_MODEL.
    
    Parameters:
        data_root: string
        - Root path to the image/patch data.
    """
    
    # Optional transforms that normalize and augment the image patches.
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    
    train_set = PatchNpyDataset(root = os.path.join(data_root, 'patch-set', 'npy', 'train'), transform = train_tf)
    val_set   = PatchNpyDataset(root = os.path.join(data_root, 'patch-set', 'npy', 'val'), transform = val_tf)
    
    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
    val_loader   = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = True)
    
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device     = torch.device(device_str)
    
    A         = np.load('a.npy')
    mmac_cnn  = MMAC_MODEL(A, 32).to(device)
    optimizer = Adam(mmac_cnn.parameters(), lr = LR_START)
    
    # We reduce the LR by a factor of 10 whenever validation error increases.
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=LR_PATIENCE, min_lr=1e-8)
    
    print(f'Training set   : {len(train_set)} samples')
    print(f'Validation set : {len(val_set)} samples')
    print(f'Labels     : {train_set.get_labels()}')
    print(f'Device     : {device_str}')
    print(f'Model type : {MMAC_MODEL}')
    
    # Prepare stuff
    train_losses = []
    val_losses  = []
    min_loss = float('inf') # The least validation loss seen so far
    
    # Training / testing loop
    for epoch in range(EPOCHS):
        lr = optimizer.param_groups[0]['lr']
        print(f'\nEPOCH {epoch}, lr = {lr}')
        
        train_loss, train_acc, train_ucost = train(mmac_cnn, device, train_loader, optimizer)
        train_losses.append(train_loss)
        
        val_loss, val_acc, val_ucost = validate(mmac_cnn, device, val_loader)
        val_losses.append(val_loss)
        
        # If this was the best-performing epoch so far,
        # save the MMAC model state
        if(val_loss < min_loss):
            min_loss = val_loss
            print('Saving mmac_cnn state to mmac_cnn_3.pt...')
            torch.save(mmac_cnn, 'mmac_cnn_3.pt')
        
        scheduler.step(val_loss)
        
    
    # Plot the losses after 
    x = np.arange(0, EPOCHS, 1)
    plt.plot(x, train_losses)
    plt.plot(x, val_losses)
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Losses')
    plt.show()
    
    
    
if __name__ == '__main__':
    # Launch program using python train_mac.py <data_root>
    #
    # The program expects the following directory layout will be used (should be default):
    # - Training image patches: <data_root>/patch-set/npy/train
    # - Validation image patches: <data_root>/patch-set/npy/val
    #
    # The program expects A.npy will be in the current working directory.
    # The program will save mmac_cnn.pt to the current working directory.
    try:
        dp = sys.argv[1]
    except:
        print('Error retreiving data root path. Run the script as "python train_mac.py <data_root>".')
        quit()

    main(dp)

    
    