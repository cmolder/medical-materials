import os
import glob
import numpy as np
import matplotlib.image as mpimg
import torch
from torch.utils.data import Dataset



class PatchPngDataset(Dataset):
        

    def __init__(self, root, transform=None):
        """Loads patch .png files to be interpreted by PyTorch.

        Parameters:
            root: string
                - Directory containing the image patches
                - "mask" and "nonmask" subdirectories expected
            
            transform : callable, optional
                - Optional transform to be applied to the sample.
        """
        self.root      = root      # Root directory of the image patches
        self.transform = transform
        self.files     = []        # Array of tuples (label, [filenames])
        prev_dir = os.getcwd()
        
        # Folder names represent the labels
        for label in [os.path.basename(f) for f in glob.glob(self.root + '\\*')]:
            os.chdir(self.root + f'\\{label}')
            self.files.append(([os.path.basename(f) for f in glob.glob(self.root + f'\\{label}\\*.png')], label))
                
        # Return directory to original working directory
        os.chdir(prev_dir)
        
        

    def __len__(self):
        """Supports array-like length interpretation of the dataset,
        getting the number of image names in the dataset.
        (This does NOT count the labels as entries, only the sub-arrays)
        
        Returns:
            length: integer
        """
        label_sizes = [np.size(label[0]) for label in self.files]
        return(np.sum(label_sizes))
    
    
    def __str__(self):
        """Prints the dataset as a string (may be bad for large datasets) 
        
        Returns:
            string: string
        """
        return str(self.files)
     
        

    def __getitem__(self, index):
        """Supports array-like indexing i.e. data[i].
            
        Parameters:
            index: int 
                - index of what image that is to be accessed
        
        Returns: 
            (image, label): (numpy array, string) tuple
        """
        if torch.is_tensor(index):
            index = index.tolist()
            
        count     = 0
        label     = -1
        label_len = 0
        
        while(count <= index):
            label     = label + 1
            label_len = len(self.files[label][0])
            count     = count + label_len
            
        image_name = self.files[label][0][index - count + label_len]
        label_name = self.files[label][1]   
        image_path = self.root + f'\\{label_name}\\{image_name}'
       
        ''' TODO Probably shouldnt use matplotlib '''
        ''' TODO apply transform to the image     '''
        image = mpimg.imread(image_path)
        image = torch.from_numpy(image)
        label = torch.tensor(label)
        
        return image, label
            
    
        # TODO should I go ahead and return the raw image data or just
        # a reference to its name?
        #
        # if self.transform:
        #     image = self.transform(image)
        # should this return a dictionary or just the image, label as a tuple?
        # return (label, image)
    
    
   
    def get_labels(self):
        """Obtains the set of label names from the files in the dataset.
        
        Returns:
            labels: [string, string, ...] array
        """
        labels = [label[1] for label in self.files]
        return labels
        

    
    def label_size(self, label):
        """Obtains the amount of entries associated with a given label
        in the dataset.
        
        Paramaters:
            label: string
        Returns: int
                - The amount of entries associated with that label.
                If the label is not present, returns 0.
        """
        for label_group in self.files:
            if label == label_group[1]:
                return np.size(label_group[0])
        return 0



    def label_to_abs_index(self, index, label):
        """Obtains the absolute index (used by data[i] notation) of an object in the 
        dataset based on the index within the label it is associated in.
        
        Parameters:
            label: string
            index: int
        Returns:
            (image, label): (numpy array, string) tuple
        """
        count     = 0
        label_found = False
        
        for i in range(0, len(self.files)):
            if (label == self.files[i][1]):
                label_found = True
                break
            else:
                count = count + len(self.files[i][0]) - 1
            
        if label_found is False:
            return -1
        else:
            return count + index
    