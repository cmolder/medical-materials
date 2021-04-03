import os, sys

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from tqdm.autonotebook import tqdm
from torch import optim
from torch.utils.data import DataLoader

from datasets import PatchNpyDataset, PatchCompare
from d_net import D_CNN


BATCH_SIZE = 50 # Number of ref/comparison groups per batch
EPOCHS     = 15 # Number of epochs

def train(model, device, train_loader, optimizer, verbose=True):
    """Trains the D-CNN for one epoch.
    
    Parameters:
        model: D_CNN
            The D-CNN neural network instance being trained.
        device: string
            The device (cpu or cuda) that the model is being run on.
        train_loader: DataLoader
            The training data.
        optimizer: torch.optim
            The optimizer used to modify the weights of the model.
        verbose: bool (optional)
            If True, prints out information (in addition to progress 
            bars) from the function.
    
    Returns:
        total_loss: float
            The total loss accured while training the D-CNN.
    """
    model.train()
    correct   = 0
    incorrect = 0
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, unit=' training batches')):        
        images     = batch[0].to(device) # Images
        cmp_labels = batch[2].to(device) # Comparison labels  
        
        optimizer.zero_grad()
        results = model(images)
        
        # Split all of the batch values into tuples of each of the (n + 1) images.
        results = torch.split(results, 1, dim=1)
        results = [res.squeeze() for res in results]
        
        cmp_labels = torch.split(batch[2], 1, dim=1)
        cmp_labels = [lbl.squeeze() for lbl in cmp_labels]
        cmp_labels = [lbl.type(torch.LongTensor).to(device) for lbl in cmp_labels]
        
        
        # Iterate through each image result in the results and backprop the loss.
        # Start at index 1 because we don't want to calc loss of reference to itself.
        #
        # TODO is summing the losses like this okay? Or do I need to go back
        # and backprop the losses individually with retain_graph = True?
        losses = []
        for i in range(1, len(results)): # TODO Check why indexes start at 1.  
            losses.append(F.cross_entropy(results[i], cmp_labels[i]))
            
        loss = sum(losses)
        total_loss += float(loss)
        loss.backward()
        optimizer.step()
            
        # print(f'Results {len(results)} x {results[0].size()}')
        # print(f'Clabs   {len(cmp_labels)} x {cmp_labels[0].size()}')
            
        # Go through the results and determine NN accuracy
        for i in range(len(cmp_labels[0])):
            for j in range(1, len(cmp_labels)):
                comp_cmp_lbl = cmp_labels[j][i].item()
                pred = torch.argmax(results[j][i]).item()
                                            
                is_correct = (pred == comp_cmp_lbl)
                
                if is_correct:
                    correct += 1
                else:
                    incorrect += 1
                    
    accuracy = correct/(correct+incorrect)
                    
    del cmp_labels # Free CUDA memory
    del results
    
    if verbose:
        print(f'Training set : {correct:5}/{correct+incorrect:5}, {accuracy*100:.2f}%')
        print(f'Training loss: {total_loss}')
    return total_loss, accuracy
    


def validate(model, device, val_loader, num_classes, verbose=True):
    """Runs the validation set on the DCNN, generating
    a D matrix in the process.

    Parameters:
        model: D_CNN
            The D-CNN neural network instance being validated.
        device: string
            The device (cpu or cuda) that the model is being run on.
        val_loader: DataLoader
            The validation data.
        verbose: bool (optional)
            If True, prints out information (in addition to progress 
            bars) from the function.
            
    Returns:
        D: np.array
            
        total_loss: float
            The total loss accured while evaluating the D-CNN on the validation set.
        accuracy: float
            The ratio of correct similarity predictions out of all predictions.
        model.state_dict(): 
            Weights of the trained D-CNN model (may be saved if it yields the lowest validation loss).
    """
    model.eval()
    correct   = 0
    incorrect = 0
    total_loss = 0.0
    
    # Used to generate the D matrix for the DCNN.
    Ref_ICs = [] # [# image sets]              List that holds the image class of each ref image viewed
    Sim_Ds  = [] # [# image sets, num_classes] List of arrays of binary simiarlity decisions for each ref image
                 #                             against the comparsion images                                            
    
    for batch_idx, batch in enumerate(tqdm(val_loader, unit=' validation batches')):   
        images     = batch[0].to(device) # Images
        img_labels = batch[1].to(device) # Image labels
        cmp_labels = batch[2].to(device) # Comparison labels
        results = model(images)
        
        # Split all of the batch values into tuples of each of the (n + 1) images.
        results = torch.split(results, 1, dim=1)
        results = [res.squeeze() for res in results]
        
        cmp_labels = torch.split(cmp_labels, 1, dim=1)
        cmp_labels = [lbl.squeeze() for lbl in cmp_labels]
        cmp_labels = [lbl.type(torch.LongTensor).to(device) for lbl in cmp_labels]
        
        img_labels = torch.split(img_labels, 1, dim=1)
        img_labels = [lbl.squeeze() for lbl in img_labels]
        img_labels = [lbl.type(torch.LongTensor) for lbl in img_labels]
        
        # print(f'Images  {len(images)} x {images[0].size()}')
        # print(f'Results {len(results)} x {results[0].size()}')
        # print(f'Ilabs   {len(img_labels)} x {img_labels[0].size()}')
        # print(f'Clabs   {len(cmp_labels)} x {cmp_labels[0].size()}')
        losses = []
        for i in range(1, len(results)): # TODO Check why indexes start at 1.  
            losses.append(F.cross_entropy(results[i], cmp_labels[i]))
            
        loss = sum(losses)
        total_loss += float(loss)
    
        # Go through the results and determine NN accuracy / update the values of P and N.
        for i in range(len(img_labels[0])):
            Ref_ICs.append(img_labels[0][i].item())
            Sim_D = np.zeros(num_classes).astype(int)
                         
            for j in range(1, len(cmp_labels)):
                comp_cmp_lbl = cmp_labels[j][i].item()
                comp_img_lbl = img_labels[j][i].item()
                pred = torch.argmax(results[j][i]).item()
                
                # Calculate the accuracy of the neural network
                is_correct = (pred == comp_cmp_lbl)
                
                if is_correct:
                    correct += 1
                else:
                    incorrect += 1
        
                Sim_D[comp_img_lbl] = int(pred)
            Sim_Ds.append(Sim_D)
                
    # P vector before normalization: the count of positive similarity decisions for each
    # reference image class against each of the other classes.
    # See SchNis (1)
    P = np.zeros([num_classes, num_classes]).astype(int)
    
    for i in range(len(Ref_ICs)):
        P[Ref_ICs[i]] += Sim_Ds[i]
    P = P.astype(np.float32) / np.bincount(Ref_ICs)

    # Generate D matrix
    # See SchNis (2)
    D = np.zeros([num_classes, num_classes]).astype(np.float32)
    for m in range(num_classes):
        for n in range(num_classes):
            D[m][n] = np.linalg.norm(P[m] - P[n])
            
    accuracy = correct/(correct+incorrect)
    
    if verbose:
        print(f'Validation set : {correct:5}/{correct+incorrect:5}, {accuracy*100:.2f}%')
        print(f'Validation loss: {total_loss}')
        print(f'P matrix   :\n{P}')
        # Print the D matrix (fancy)
        # print('\nD matrix   :')
        # print('       l0    l1    l2')
        # for row_name, row in zip(['l0', 'l1', 'l2'], D):
        #     line = f'{row_name} ['
        #     for i in row:
        #         line += f'{i:.3f} '.lstrip('0').rjust(6)
        #     line += ']'
        #     print(line)
        # Print the D matrix (simple)
        print(f'D matrix   :\n{np.around(D, decimals=3)}')
    
    del cmp_labels # Free CUDA memory
    del img_labels
    del results
        
    return D, total_loss, accuracy, model.state_dict()



def main(data_root):
    """Driver class for training the D-CNN.
    
    Parameters:
        data_root: string
        - Root path to the image/patch data.
    """
    # Optional transforms that normalizes the image for ResNet.
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    
    val_tf = transforms.Compose([
    ])
    
    # Load the material patch datasets
    # train_set     = PatchNpyDataset(root = f'{data_root}/patch-set/npy/train/'), transform = train_tf)
    train_set     = PatchNpyDataset(root = os.path.join(data_root, 'patch-set', 'npy', 'train'), transform = train_tf)
    train_samples = PatchCompare(train_set)
    train_loader  = DataLoader(train_samples, batch_size=BATCH_SIZE, shuffle=True)
    train_losses  = []
    train_accuracies = []
    print(f'Training set   : {len(train_samples)} samples')
    
    # val_set     = PatchNpyDataset(root = f'{data_root}/patch-set/npy/val/'), transform = val_tf)
    val_set     = PatchNpyDataset(root = os.path.join(data_root, 'patch-set', 'npy', 'val'), transform = val_tf)
    val_samples = PatchCompare(val_set)
    val_loader  = DataLoader(val_samples, batch_size=BATCH_SIZE, shuffle=True)
    val_losses  = []
    val_accuracies = []
    print(f'Validation set : {len(val_samples)} samples')
    
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device     = torch.device(device_str)
    d_cnn      = D_CNN().to(device)
    optimizer  = optim.Adam(d_cnn.parameters(), lr=1e-3)

    # max_acc = 0.0 # Maximum accuracy of the D-CNN on the validation set of all epochs so far
    min_loss   = float('inf') # Lowest loss of the D-CNN on the validation set of all epochs so far

    print(f'Labels : {train_set.get_labels()}')
    print(f'Device : {device_str}')
    
    # for i in range(len(train_samples) - 1, len(train_samples) - 20, -1):
    #    train_samples.display_group(i)
    for epoch in range(EPOCHS):
        print(f'\nEPOCH {epoch}')
        
        loss, acc = train(d_cnn, device, train_loader, optimizer)
        train_losses.append(loss)
        train_accuracies.append(acc * 100.0)
        
        D, loss, acc, model_state = validate(d_cnn, device, val_loader, len(val_samples.classes))
        val_losses.append(loss)
        val_accuracies.append(acc * 100.0)
        
        # If this is the lowest-loss (and therefore generally most accurate)
        # run so far, save the D matrix to disk
        if loss < min_loss:
            min_loss = loss
            print('\nSaving model to dcnn.pt...')
            torch.save(model_state, 'dcnn.pt')
            print('Saving D matrix to D.npy...')
            np.save('d.npy', D)
        

        
if __name__ == '__main__':
    # Launch program using python test_dcnn.py <data_root>
    #
    # The program expects the following directory layout will be used (should be default):
    # - Training image patches: <data_root>/patch-set/npy/train
    # - Validation image patches: <data_root>/patch-set/npy/val
    # The program will save dcnn.pt, D.npy to the current working directory.
    try:
        dp = sys.argv[1]
    except:
        print('Error retreiving data root path. Run the script as "python test_mac_conv.py <data_root>".')
        quit()

    main(dp)
