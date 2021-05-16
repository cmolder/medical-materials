import os, sys

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import PatchNpyDataset
from mmac_net.train_helpers import loss_acc


BATCH_SIZE = 50 # Default batch size when running from command line.

def tsne_raw(test_set):
    """Generates a t-SNE embedding for the raw feature vectors (i.e. the patch image data)
    """
    
    # Get an array of integers for the testing set
    # so we can shade our tsne-graph with the labels
    label_strs = test_set.get_labels()
    # labels     = np.arange(len(label_strs))
    test_set = list(test_set)
    
    X = np.array([i[0].numpy() for i in test_set])
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    y = np.array([i[1].numpy() for i in test_set])
    
    X_embedded = TSNE().fit_transform(X)

    tsne_plot(X_embedded, y, label_strs, title='MAC-CNN raw features')
    

def tsne_learned(test_set, A_preds):
    """Generates a t-SNE embedding for the resulting categeories learned by the MAC-CNN.
    """

    label_strs = test_set.get_labels()
    # labels     = np.arange(len(label_strs))
    test_set   = list(test_set)
    
    X = A_preds
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    y = np.array([i[1].numpy() for i in test_set])
    
    print(np.shape(X))
    
    X_embedded = TSNE().fit_transform(X)
    tsne_plot(X_embedded, y, label_strs, title='MAC-CNN A predictions')
    


def tsne_plot(embedding, labels, label_strs, title=''):
    """ A helper function for the t-SNE embeddings that plots the data provided.
    """
    label_nums = np.arange(len(label_strs))
    
    fig, ax = plt.subplots()
    for l in label_nums:
        ix = np.where(labels == l)
        ax.scatter(embedding[ix].T[0], embedding[ix].T[1], 
                   label = label_strs[l], s = 1)
    ax.legend()
    plt.title(title)
    plt.show()
        
    

def correlation_matrix(Ypreds, Afpreds, Ycats):
    """Generates a correlation matrix between true-value categories and 
    evaluated attributes.
    """
    Ypreds = Ypreds.T   # Transpose for easier caluclation
    Afpreds = Afpreds.T
    C = np.empty((np.shape(Ypreds)[0], np.shape(Afpreds)[0])) # Correlation matrix
    
    for i in range(np.shape(Ypreds)[0]):
        for j in range(np.shape(Afpreds)[0]):
            C[i][j] = np.correlate(Ypreds[i], Afpreds[j]) # TODO Is this the correct way to make correlation? Do we use y labels or predictions?

    C = C - C.mean()        # Normalize C to [-1, 1]
    C = C / np.abs(C).max()
    
    # Plot the matrix
    fix, ax = plt.subplots()
    im = ax.imshow(C, cmap='seismic')
    
    ax.set_yticks(np.arange(len(Ycats)))
    ax.set_xticks(np.arange(np.shape(Afpreds)[0]))
    ax.set_yticklabels(Ycats)
    ax.set_xticklabels(np.arange(np.shape(Afpreds)[0]))
    ax.set_ylabel('Material category')
    ax.set_xlabel('Material attribute')
    
    # Loop over the data and plot correlation values in squares
    for i in range(np.shape(Ypreds)[0]):
        for j in range(np.shape(Afpreds)[0]):
            if C[i, j] > -0.4 and C[i, j] < 0.1:
                ax.text(j, i, round(C[i, j], 2), ha='center', va='center', color='black')
            else:
                ax.text(j, i, round(C[i, j], 2), ha='center', va='center', color='white')
    
    ax.set_title('MAC-CNN correlation matrix')
    plt.show()



def test(model, device, test_loader,
          w_per = 1e0, w_kld = 1e-2):
    """Tests the MMAC-CNN.
    
    Parameters:
        model: torch.nn
            The neural network being tested.
        device: string
            The device (cpu or cuda) that the model is being run on.
        test_loader: Dataloader
            The test data.
    """
    model.eval()
    
    A = model.A
    test_loss  = 0.0
    test_acc   = 0.0
    test_ucost = 0.0
    all_Apreds = torch.empty(0, model.m, 6).to(device)
    all_Ypreds = torch.empty(0, model.k).to(device)
    
    for batch_idx, batch in enumerate(tqdm(test_loader, unit=' testing batches')):
        X, y = batch

        X = X.to(device)
        y = y.to(device)
        
        # Forward pass
        # y_pred is the k class prediction
        # A_preds are the a1, a2, ..., a5, a_final m attribute predictions
        # from each level of the auxillary layers
        y_pred, A_preds = model(X)

        # Add the A_preds from this batch to the entire set (for learned t-SNE embedding)
        A_preds_tmp = torch.stack(A_preds, dim=2)
        all_Apreds = torch.cat([all_Apreds, A_preds_tmp], dim=0)
        all_Ypreds = torch.cat([all_Ypreds, y_pred], dim=0)
        # all_Yacts  = torch.cat([all_Yacts, y.float()], dim=0)
        
        # l is the loss
        # a is the accuracy
        # u is the u-cost
        # y is a k-dimensional one-hot vector
        # A is the (k x m) matrix      
        loss, acc, u_cost = loss_acc(y, y_pred, A, A_preds, w_kld, w_per)
        
        test_loss  += float(loss)
        test_acc   += acc
        test_ucost += u_cost

    # Print results
    print()
    print(f'Test loss  : {test_loss / len(test_loader)}')
    print(f'Test ucost : {test_ucost / len(test_loader)}')
    print(f'Test acc   : {test_acc / len(test_loader) * 100:.3f}%')
    
    
    print(f'A_preds {all_Apreds.size()}')
    print(f'y_preds {all_Ypreds.size()}')
    
    return all_Ypreds.cpu().detach().numpy(), all_Apreds.cpu().detach().numpy() #, all_Yacts.cpu().detach().numpy()



def main(data_root):
    """Driver class for testing loop, to run a t-sne embedding
    and see how well the network performs.
    
    Parameters:
        data_root: string
        - Root path to the image/patch data.
    """
    with torch.no_grad():
        test_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
         
        # test_set    = PatchNpyDataset(root = f'{data_root}/patch-set/npy/test', transform = test_tf)
        test_set   = PatchNpyDataset(root = os.path.join(data_root, 'patch-set', 'npy', 'test'), transform = test_tf)
        test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False)
        
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        device     = torch.device(device_str)
        mmac_cnn   = torch.load('mmac_cnn_2.pt').to(device)
        
        print(test_set.get_labels())
        
        all_Ypreds, all_Apreds = test(mmac_cnn, device, test_loader)
        
        # Generate a correlation matrix between the attributes and categories.
        correlation_matrix(all_Ypreds, all_Apreds[:,:,-1], test_set.get_labels())
        
        # Use t-sne embedding to see how well MMAC's A attributes separate
        # the k material categories
    
        tsne_learned(test_set, all_Apreds)
        # test_set = PatchNpyDataset(root = f'{data_root}/patch-set/npy/test') # Reinit test set b/c pillow transform makes it unworkable with tsne_raw
        test_set = PatchNpyDataset(root = os.path.join(data_root, 'patch-set', 'npy', 'test')) # Reinit test set b/c pillow transform makes it unworkable with tsne_raw
        tsne_raw(test_set)


# VGG-16         : 93.385% (93.4%)
# ResNet34       : 92.816% (92.8%)
# ResNet34 (no A): 91.738% (91.7%)
    
if __name__ == '__main__':
    # Launch program using python test_dcnn.py <data_root>
    #
    # The program expects the following directory layout will be used (should be default):
    # - Test image patches: <data_root>/patch-set/npy/test
    # The program also expects mmac_cnn.pt to be in the current directory.
    try:
        dp = sys.argv[1]
    except:
        print('Error retreiving data root path. Run the script as "python test_mac_conv.py <data_root>".')
        quit()

    main(dp)

    
