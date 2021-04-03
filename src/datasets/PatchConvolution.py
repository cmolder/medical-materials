import numpy as np
import torch

from numpy.random import choice, shuffle
from torch.utils.data import Dataset
from matplotlib import colors
import matplotlib.pyplot as plt



class PatchConvolution(Dataset):
    

    def __init__(self, imageset, index, stride=1, size=32, transform=None):
        """Dataset that takes in exactly one medical image from a loader and 
        on each iteration returns a convolutional slice of the image as a patch.

        Used to generate per-pixel material category/attribute predictions
        using a sliding-window basis.

        This does NOT depend on any of the Patch*.py files, despite its name.
        It only takes in images from a Loader class. There is no need to
        generate patches from the imageset loader beforehand, either.

        The dataset does not consider patch labels since it is meant to be
        used only for testing the MMAC against the image/mask.

        Parameters:
            imageset: Loader
            - The set of loaded images from which to generate convolutions.
            index: int
            - The particular image in the imageset that will be used to
              generate convolutions.
            stride: int
            - Convolutional stride
            size: int
            - n x n size convolutions (image patches)
        """
        self.image, self.mask = imageset[index]
        if imageset.masked:
            self.mask   = self.mask.astype(np.float32)
            self.masked = True
        else:
            self.masked = False
        
        # Pad image to fit the stride
        orig_img   = self.image.astype(np.float32)
        orig_shape = np.shape(orig_img)
        print(f'Original image shape: {orig_shape}')
        
        new_x = orig_shape[0] + (stride - orig_shape[0] % stride)
        new_y = orig_shape[1] + (stride - orig_shape[0] % stride)
        print(f'Padded image shape: ({new_x}, {new_y})')
        
        self.image = np.zeros([new_x, new_y], dtype=np.float32)
        self.image[:orig_shape[0], :orig_shape[1]] = orig_img
        
        
        self.stride    = stride
        self.size      = size
        self.transform = transform
        
        

    def __len__(self):
        """Supports array-like length interpretation of the dataset,
        getting the number of convolutional slices from the image.
        
        Returns:
            length: integer
        """
        num_x, num_y = self.conv_dims()
        return num_x * num_y
    
    

    def __getitem__(self, index):
        """Supports array-like indexing i.e. data[i]. Does not get the mask
        data, if that is provided with the image.
            
        Parameters:
            index: int 
            - index of what image that is to be accessed
        
        Returns: 
            image: Numpy array of size*size
            - The ith convolutional slice from the image.
        """
        x, y  = self.location_of(index)

        # Do not take patches from out of bounds spaces
        #if x > img_w - self.size or y > img_h - self.size:
        #    raise IndexError('Patch boundary out of bounds')
        
        patch = torch.from_numpy(self.image[x : x + self.size, y : y + self.size])
        if np.shape(patch) != torch.Size([32, 32]):
            print(f'({x}, {y}) {np.shape(patch)}')
        if self.transform:
            patch = self.transform(patch)
        return patch
    
    
    
    def get_mask(self, index):
        """Analogous to __getitem__, but for the mask patch IF it is masked.
        """
        if not self.masked:
            return None
        
        x, y  = self.location_of(index)
        # Do not take patches from out of bounds spaces
        #if x > img_w - self.size or y > img_h - self.size:
         #   raise IndexError('Patch boundary out of bounds')
        
        patch = torch.from_numpy(self.mask[x : x + self.size, y : y + self.size])
        
        if self.transform:
            patch = self.transform(patch)
        return patch
    
    
    
    def location_of(self, index):
        """Gets the (x, y) location of the top-left corner of the
        (index)th convolutional slice.
    
        Parameters:
            index: int 
                - Index of what image that we are trying to get
                the location of
        
        Returns: 
            x, y: int, int
                - (x, y) location of the top-left corner of
                the patch.
        """
        img_w, img_h = self.conv_dims()
    
        x = (index % img_w) * self.stride
        y = (index // img_w) * self.stride
        
        return x, y



    def conv_dims(self):
        """Gets the dimensions of the convolution locations.
        """
        img_w = np.shape(self.image)[0]
        img_h = np.shape(self.image)[1]
        
        x = (img_w - self.size) // self.stride
        y = (img_h - self.size) // self.stride
        
        return x, y


    def display(self, index, mask_overlay = False, save = False):
        """Displays the ith convolutional slice using MatPlotLib.
        
        Parameters:
            index: int 
                - Index of what image that we are trying to display
            mask_overlay: boolean
                - If the set has a corresponding mask (masked = true),
                overlay the mask on top of the patch.
            save: boolean
                - If true, then save the plot to disk.
        """
        image_data = self[index].squeeze().numpy()
        x, y       = self.location_of(index)
            
        plt.figure(index)
        plt.title(f'Image {index} at ({x}, {y})')
        plt.imshow(image_data, cmap='bone')
        
        if save:
            plt.savefig(f'{index}img.png', dpi = 400)
        
        if self.masked and mask_overlay:
            mask_data = self.get_mask(index).squeeze().numpy()
            plt.imshow(mask_data, cmap='viridis', alpha=0.4)
            
            if save:
                plt.savefig(f'{index}mask.png', dpi = 400)
        
        plt.show()