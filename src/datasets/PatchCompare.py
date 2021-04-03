import numpy as np
import torch

from numpy.random import choice, shuffle
from torch.utils.data import Dataset
from matplotlib import colors
import matplotlib.pyplot as plt


class PatchCompare(Dataset):
    
    def __init__(self, dataset, classes = None, samples = None):
        """Dataset that on each iteration returns a set of patches -
        one patch is the reference patch with a reference class,
        and the other patches are one patch of each class in the dataset.

        Parameters:
            dataset: PatchDataset
                - The patch dataset from which to generate pairs
            classes: [string, string, ...]
                - The string names for the particular classes to be chosen
                - If none provided, defaults to (MIN OF) all the classes in dataset
            samples: int
                - The number of comparison groups to generate for each class.
                - If none provided, defaults to one per images in label.
        """
        self.dataset  = dataset
        
        # List of tuples of the form (ref_patch, [patches])
        # patches[i] are shuffled, but each class gets one patch
        # in the list. 
        #
        # For reference, each patch is in the form (image, label)
        self.groups = []
        
        if classes == None:
            self.classes = dataset.get_labels()
        else:
            self.classes = classes
            
        class_sizes = [self.dataset.label_size(class_name) for class_name in self.classes]

        if samples is None:
            samples = min(class_sizes)
        
        for i, cl in enumerate(self.classes):
            
            # Dataset indexes for the reference images in the dataset.
            ref_idxs = choice(class_sizes[i], size = samples, replace = False)
            ref_idxs = [self.dataset.label_to_abs_index(idx, cl) for idx in ref_idxs]
    
            # For each reference image, generate a list of comparison images - one
            # from each class (including the reference image class).
            for j in range(samples):
                comp_idxs = [choice(class_sizes[k]) for k in range(len(self.classes))]        
                comp_idxs = [self.dataset.label_to_abs_index(comp_idxs[k], self.classes[k]) for k in range(len(self.classes))]
                self.groups.append((ref_idxs[j], comp_idxs))
                           
                
          
    def __len__(self):
        """Overrides the length for the dataset.
    
        Returns: int
            - Length of the dataset.
        """
        return len(self.groups)
    
    
    
    def __getitem__(self, index):
        """Overrides array indexing for the dataset.
        
        Returns: 
            (image, comp_label, image_label) tuple
            - image is a (classes + 1) * (img_size) * (img_size) tensor of images
            for the comparison group. The 0th entry is the reference image and
            the other images are being compared to it.
            - image_label is a (class + 1) one-dimensional tensor of the labels
            for the images.
            - comp_label is a (class + 1)-long one-hot tensor. If the image's label
            is the same as the reference image's label, it equals 0. Otherwise
            it equals 1.
        """
        if torch.is_tensor(index):
            index = index.tolist()
            
        images     = torch.zeros([len(self.classes) + 1, 32, 32])
        cmp_labels = torch.zeros(len(self.classes) + 1)
        img_labels = torch.zeros(len(self.classes) + 1)
        
        ref_img       = self.dataset[self.groups[index][0]]
        images[0]     = ref_img[0]
        img_labels[0] = ref_img[1]
        cmp_labels[0] = torch.tensor(0)


        comps = self.groups[index][1]
        shuffle(comps)
        
        for i, comp in enumerate(comps):
            comp_img = self.dataset[comp]
            images[i + 1]     = comp_img[0]
            img_labels[i + 1] = comp_img[1]     
            cmp_labels[i + 1] = torch.tensor(0) if img_labels[i+1] == img_labels[0] else torch.tensor(1)
            
        return images, img_labels, cmp_labels


    def display_group(self, index):
        """Plots one sample at index (index),
        with the reference patch and comparison patches.

        Parameters:
            index: int
            - The index of the group to be displayed.
        """
        images, img_labels, cmp_labels = self[index]
        cmap = 'plasma'
        norm = colors.Normalize(vmin = 0, vmax = 1)

        
        ref_image   = images[0].numpy()
        ref_img_lbl = img_labels[0].item()
        ref_cmp_lbl = cmp_labels[0].item()
        
        comp_images = []
        comp_img_lbls = []
        comp_cmp_lbls = []
        
        for i in range(1, len(images)):
            comp_images.append(images[i].numpy())
            comp_img_lbls.append(img_labels[i].item())
            comp_cmp_lbls.append(cmp_labels[i].item())
            
        fig, axs = plt.subplots(1, len(comp_images) + 1)
        # fig.suptitle(f'Patch comparison group {index}')	
        
        images   = []
    
        # First image: reference patch
        images.append(axs[0].imshow(ref_image, cmap = cmap))
        axs[0].set_title(f'REF: img {int(ref_img_lbl)}, cmp {int(ref_cmp_lbl)}')
        axs[0].axis('off')
        
        for i in range(len(comp_images)):
            images.append(axs[i + 1].imshow(comp_images[i], cmap = cmap))
            axs[i + 1].set_title(f'img {int(comp_img_lbls[i])}, cmp {int(comp_cmp_lbls[i])}')
            axs[i + 1].axis('off')
            
        # Set the colorbar
        
        for i in images:
            i.norm = norm
        
        plt.tight_layout()
        fig.colorbar(images[0], ax = axs, orientation='vertical', shrink = 0.5, aspect = 15)
        plt.show()
            