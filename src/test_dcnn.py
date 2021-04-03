import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import PatchNpyDataset, PatchCompare
from d_net import D_CNN


BATCH_SIZE = 50  # Number of pos/neg pairs per batch


def accuracy_matrix(Acc, labels):
    """Plots an accuracy matrix between similarity decisions of patches from class i to j,
    where the first axis is the reference class and the second axis the comparison class.
    
    Parameters:
        Acc: numpy array
            Accuracy matrix to be plotted
        labels: list
            List of strings representing the categories compared
    """
    fig, ax = plt.subplots()
    im = ax.imshow(Acc, cmap='seismic')
    
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Reference category')
    ax.set_xlabel('Comparison category')
    
    # Loop over the data and plot accuracy values in squares
    for i in range(np.shape(Acc)[0]):
        for j in range(np.shape(Acc)[0]):
            if Acc[i, j] > 0.75 and Acc[i, j] < 0.875:
                ax.text(j, i, round(Acc[i, j], 3), ha='center', va='center', color='black')
            else:
                ax.text(j, i, round(Acc[i, j], 3), ha='center', va='center', color='white')
                
    ax.set_title('D-CNN accuracy matrix')
    plt.show()
    
    

def test(model, device, test_loader, num_classes):
    """Tests the DCNN.
    
    Parameters:
        model: torch.nn
            The neural network being tested.
        device: string
            The device (cpu or cuda) that the model is being run on.
        test_loader: Dataloader
            The test data.
            
    Returns:
        Acc: np.array
            The accuracy matrix (TODO check if it is by class?)
    """
    model.eval()
    correct   = 0
    incorrect = 0
    
    # Used to generate the D matrix for the DCNN.
    Ref_ICs = [] # [# image sets]              List that holds the image class of each ref image viewed
    Sim_Ds  = [] # [# image sets, num_classes] List of arrays of binary simiarlity decisions for each ref image
                 #                             against the comparsion images                  
                 
    correct_mtx = np.zeros((num_classes,num_classes)) # Number of times DCNN is correct when comaparing category i to j
    counts_mtx  = np.zeros((num_classes,num_classes)) # Number of times comparisons of category i to j occur
    
    for batch_idx, batch in enumerate(tqdm(test_loader, unit=' testing batches')):   
        images     = batch[0].to(device) # Images
        img_labels = batch[1].to(device) # Image labels      -> value is true class of each of the (n+1) patches
        cmp_labels = batch[2].to(device) # Comparison labels -> values are if this patch is same class as reference patch (index 0)
        results = model(images)
        
        # Split all of the batch values into tuples of each of the (n + 1) images.
        results = torch.split(results, 1, dim=1)
        results = [res.squeeze() for res in results]
        
        cmp_labels = torch.split(cmp_labels, 1, dim=1)
        cmp_labels = [lbl.squeeze() for lbl in cmp_labels]
        cmp_labels = [lbl.type(torch.LongTensor) for lbl in cmp_labels]
        
        img_labels = torch.split(img_labels, 1, dim=1)
        img_labels = [lbl.squeeze() for lbl in img_labels]
        img_labels = [lbl.type(torch.LongTensor) for lbl in img_labels]
        
        # print(f'Images  {len(images)} x {images[0].size()}')
        # print(f'Results {len(results)} x {results[0].size()}')
        # print(f'Ilabs   {len(img_labels)} x {img_labels[0].size()}')
        # print(f'Clabs   {len(cmp_labels)} x {cmp_labels[0].size()}')
    
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
                
                # Record which two classes are being compared in the
                # counts matrix. 
                counts_mtx[Ref_ICs[-1]][comp_img_lbl] += 1
                
                
                if is_correct:
                    correct += 1
                
                    # Do the same with the correct matrix.
                    correct_mtx[Ref_ICs[-1]][comp_img_lbl] += 1
                else:
                    incorrect += 1
        
                Sim_D[comp_img_lbl] = int(pred)
            Sim_Ds.append(Sim_D)
                
                
    # Accuracy matrix by reference class (priamry axis) to comparison class (secondary axis)
    Acc = correct_mtx/counts_mtx
    
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
            
    print(f'Correct matrix : \n{correct_mtx}')
    print(f'Counts matrix  : \n{counts_mtx}')
    print(f'Accuracy matrix: \n{np.around(Acc, decimals=3)}')
    print(Acc[0][3])
    
    accuracy = correct/(correct+incorrect)
    print(f'Testing set: {correct:5}/{correct+incorrect:5}, {accuracy*100:.2f}%')
    print(f'P matrix   :\n{P}')
    print(f'D matrix   :\n{np.around(D, decimals=3)}')
    
    return Acc



def main(data_root):
    """Driver class for testing loop

    Parameters:
        data_root: string
        - Root path to the image/patch data.
    """
    
    # Load the testing set
    test_set     = PatchNpyDataset(root = os.path.join(data_root, 'patch-set', 'npy', 'test'))
    test_labels  = test_set.get_labels()
    print(f'Set labels: {test_set.get_labels()}')
    test_samples = PatchCompare(test_set)
    test_loader  = DataLoader(test_samples, batch_size = BATCH_SIZE, shuffle = True)
    
    # Load the DCNN
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device     = torch.device(device_str)
    dcnn       = D_CNN().to(device)
    dcnn.load_state_dict(torch.load('dcnn.pt'))
    
    Acc = test(dcnn, device, test_loader, num_classes = len(test_samples.classes))
    accuracy_matrix(Acc, test_labels)
    
    
    
if __name__ == '__main__':
    # Launch program using python test_dcnn.py <data_root>
    #
    # The program expects the following directory layout will be used (should be default):
    # - Test image patches: <data_root>/patch-set/npy/test
    # The program also expects dcnn.pt to be in the current directory.
    try:
        dp = sys.argv[1]
    except:
        print('Error retreiving test patch set path. Run the script as "python test.py <data_root>".')
        quit()

    main(dp)
