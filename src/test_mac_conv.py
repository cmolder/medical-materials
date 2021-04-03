import os, sys

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2

from loaders import *
from datasets import PatchConvolution, PatchNpyDataset

BATCH_SIZE  = 50
NUM_SAMPLES = 10
STRIDE      = 4

def test(model, device, test_loader,
         return_Af = True, return_Y = True):
    """Runs the MMAC-CNN on the provided Dataloader, for the
    purpose of generating a list of predictions only (no loss calculated).
    
    Parameters:
        model: torch.nn
            The neural network being tested.
        device: string
            The device (cpu or cuda) that the model is being run on.
        test_loader: DataLoader
            The test data.
        save: boolean
            If True, it saves the best epoch so far to disk. 
    """
    model.eval()
    
    # A_final material attribute predictions
    all_Afpreds = torch.empty(0, model.m).to(device) if return_Af else None

    # k material category predictions
    all_Ypreds = torch.empty(0, model.k).to(device) if return_Y else None
    
    for batch_idx, batch in enumerate(tqdm(test_loader, unit=' testing batches')):
        X = batch
        X = X.to(device)
        
        # Forward pass
        # y_pred is the k class prediction
        # A_preds are the a1, a2, ..., a5, a_final m attribute predictions
        # from each level of the auxillary layers
        y_pred, A_preds = model(X)
        
        if return_Af:
            Af_pred = A_preds[-1] # Get the final A prediction from the A_preds
            all_Afpreds = torch.cat([all_Afpreds, Af_pred], dim=0)
            del Af_pred
            
        if return_Y:
            all_Ypreds = torch.cat([all_Ypreds, y_pred], dim=0)
            
        del y_pred, A_preds
    
    return all_Ypreds, all_Afpreds


def plot_preds(image, conv_data, labels, data_root, mask=None, title=None, save=False):
    """Plots the convolution's predictions alongside the default image 
    and the mask (if present)
    
    Parameters:
        image: numpy array
            The image data that is being evaluated.
        conv_data: numpy array
            The results from convolving the image about patches to determine
            category.
        labels: dictionary
            A list of index-value pairs that map argmaxed values in the conv_data
            to actual labels in the testing set.
        mask: numpy array (optional)
            The ground-truth mask that separates categories in actuality, not based
            on the MMAC-CNN's analysis.
        title: string (optional)
            A title for the plot, to keep track of image number/plot type/etc.
    """
    cvals    = np.arange(len(labels)).tolist()
    
    fig, axs = plt.subplots(1, 3)
    
    cmapl = plt.cm.plasma # Label / classificaiton colormap
    cmapi = plt.cm.bone   # Raw image colormap
    norm    = mcolors.Normalize(vmin = min(cvals), vmax = max(cvals))
    patches = [mpatches.Patch(color=cmapl(norm(i)), label=labels[i].format(l=cvals[i])) for i in range(len(cvals)) ]
    
    # Plot the raw conv data in the third subplot
    axs[2].imshow(conv_data, cmap=cmapl, norm=norm, interpolation=None)
      
    # Plot the conv data over the image in the second subplot
    axs[1].imshow(image, cmap=cmapi)
    conv_data = conv_data.astype(np.uint8)
    conv_data = cv2.resize(conv_data, dsize=np.shape(image), interpolation=cv2.INTER_LINEAR)
    axs[1].imshow(conv_data,  norm=norm, cmap=cmapl, alpha=0.5)
    
    # Plot the raw image + mask (if present) in the first subplot
    axs[0].imshow(image, cmap=cmapi)
    if mask is not None:
        axs[0].imshow(mask, norm=norm, cmap=cmapl, alpha=0.5)
        
    # Plot the title
    if title is not None:
        plt.title(title)
        
    # Plot the legend
    plt.legend(handles = patches, bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0. )
    fig.tight_layout()
    # plt.show()
    if title is not None and save is True:
        # plt.savefig(f'{data_root}/conv-figs/{title}.png', dpi=192, bbox_inches='tight')
        plt.savefig(os.path.join(data_root, 'conv-figs', f'{title}.png'), dpi=192, bbox_inches='tight')
    


def plot_attrs(image, conv_data, data_root, title = None, mask = None, save = False):
    """Plots a heatmap for a given category alongside the default image
    and the mask (if present)
    
    Parameters:
        image: numpy array
            The image data that is being evaluated.
        conv_data: numpy array
            The results from convolving the image about patches to
            determine how strongly a given patch correaltes with
            each of the attributes.
        mask: numpy array (optional)
            The ground-truth mask that separates categories in actuality, not based
            on the MMAC-CNN's analysis.
        title: string (optional)
            A title for the plot, to keep track of image number/plot type/etc. 
    """
    M = np.shape(conv_data)[2]
    
    fig, axs = plt.subplots(M, 3)
    
    cmapl = plt.cm.plasma # Attribute intensity colormap
    cmapi = plt.cm.bone   # Raw image colormap
    norm  = mcolors.Normalize(vmin = np.amin(conv_data), vmax = np.max(conv_data))
    
    # For each attribute, plot the three relevant items: 
    # - raw image + mask (if provided)
    # - raw image + attribute heatmap
    # - raw attribute heatmap
    for m in range(M):
        # First column: raw image + mask (if provided)
        axs[m, 0].imshow(image, cmap=cmapi)
        axs[m, 0].axis('off')
        if mask is not None:
            axs[m, 0].imshow(mask, norm=mcolors.Normalize(vmin = 0, vmax = 1), cmap=cmapl, alpha=0.5)
            
        # Second column: raw image + attribute heatmap
        axs[m, 1].imshow(image, cmap=cmapi)
        axs[m, 1].axis('off')
        overlay = conv_data[:,:,m]
        overlay = cv2.resize(overlay, dsize=np.shape(image), interpolation=cv2.INTER_LINEAR)
        axs[m, 1].imshow(overlay, norm=norm, cmap=cmapl, alpha=0.5)
        
        # Third column: raw attribute heatmap
        axs[m, 2].imshow(conv_data[:,:,m], cmap=cmapl, norm=norm, interpolation=None)
        axs[m, 2].axis('off')
        
    # Plot the title
    if title is not None:
        axs[0,1].title.set_text(title)
        
    # Plot the colorbar
    fig.colorbar(cm.ScalarMappable(norm = norm, cmap = cmapl), ax = axs, anchor=(1, 0))
    # fig.tight_layout()
    
    # If set to save, then save
    if title is not None and save is True:
        # plt.savefig(f'{data_root}/conv-figs/{title}.png', dpi=288, bbox_inches='tight')
        plt.savefig(os.path.join(data_root, 'conv-figs', f'{title}.png'), dpi=288, bbox_inches='tight')



def main(data_root):
    """Driver class for a second testing loop, which attempts
    to convolve patches around a single medical image to
    determine material attributes/categories on a per-pixel
    basis.

    Parameters:
        data_root: string
        - Root path to the image/patch data.
    """
    test_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    
    
    # Load CHECK images
    # test_images = DcmLoader(root=f'{data_root}/check-data/test', label='bone', masked=False)
    
    # Load BTP images
    # bt-data-conv is a set of BTP images isolated from training/test/val patching to ensure
    # that the program has not seen these images before or any potential patches from it
    # test_images = MatLoader(root = f'{data_root}/bt-data-conv', label='brain', mask_label='tumor', masked=True)
    # patch_set  = PatchNpyDataset(root = f'{data_root}/patch-set/npy/test') # Use this to get the y_labels

    test_images = MatLoader(root = os.path.join(data_root, 'bt-data-conv'), label='brain', mask_label='tumor', masked=True)
    patch_set  = PatchNpyDataset(root = os.path.join(data_root, 'patch-set', 'npy', 'test')) # Use this to get the y_labels

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device    = torch.device(device_str)
    mac_cnn   = torch.load('mmac_cnn.pt').to(device)

    # Get label categories
    y_labels = patch_set.get_labels()
    A_labels = np.arange(mac_cnn.m).tolist()
    
    # Sample 10 random images from the test_images set and run the
    # MMAC convolutionally on them
    idxs = np.random.randint(0, len(test_images.images) + 1, NUM_SAMPLES)
    print(f'Random indexes selected:\n{idxs}')
    
    for index in idxs:
        conv_set    = PatchConvolution(test_images, index, stride = STRIDE, transform = test_tf)
        conv_loader = DataLoader(conv_set, batch_size = BATCH_SIZE, shuffle = False)

        # Evaluate the convolution set on the MAC-CNN
        all_Ypreds, all_Apreds = test(mac_cnn, device, conv_loader, return_Af = True, return_Y = True)
        # print(np.shape(all_Apreds))
        # print(np.shape(all_Ypreds))
        
        # Post-process Y prediction data to be graphable
        all_Ypreds = torch.argmax(all_Ypreds, dim=1)
        all_Ypreds = torch.reshape(all_Ypreds, conv_set.conv_dims()).detach().cpu().numpy()
        all_Ypreds = np.rot90(all_Ypreds)
        all_Ypreds = np.flipud(all_Ypreds)
        
        
        A_dims =  *conv_set.conv_dims(), mac_cnn.m
        # print("a dims:")
        # print(A_dims)
        # Post-process A prediction data to be graphable
        all_Apreds = all_Apreds.detach().cpu().numpy()
        all_Apreds = np.reshape(all_Apreds, A_dims)
        all_Apreds = np.rot90(all_Apreds)
        all_Apreds = np.flipud(all_Apreds)
        
        # print(np.shape(all_Apreds))
        timg, tmask = test_images[index]
        plot_preds(timg, all_Ypreds, y_labels, data_root, mask=tmask, title=f'{index} y', save=True) # Plot y predictions
        plot_attrs(timg, all_Apreds, data_root, mask=tmask, title=f'{index} A', save=True) # Plot A predictions
        
        del all_Ypreds, all_Apreds, conv_set, conv_loader
    
    
    
if __name__ == '__main__':
    # Launch program using python test_dcnn.py <data_root>
    #
    # The program expects the following directory layout will be used (close to default). That is:
    # - Whole "Brain-Tumor-Progression" images: <data_root>/bt-data-conv
    #    - Intended to be separted from the other BT images, to remove potential impacts from training on the set.
    # - Whole "CHECK" images: <data_root>/check-data/test
    #    - OK to use testing set here, since the testing set is not used in training/validation.
    # - Test image patches: <data_root>/patch-set/npy/test
    # 
    # The program will save the whole-image convolution graphs to:
    # - <data_root>/conv-figs/*.png
    try:
        print(sys.argv)
        dp = sys.argv[1]
    except:
        print('Error retreiving data root path. Run the script as "python test_mac_conv.py <data_root>".')
        quit()

    main(dp)
