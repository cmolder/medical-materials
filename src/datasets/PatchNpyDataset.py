import os
import glob
import numpy as np
import matplotlib.image as mpimg
import torch
from torch.utils.data import Dataset


class PatchNpyDataset(Dataset):
        


    def __init__(self, root, transform=None, sampler=None):     
        """Loads patch .npy files to be interpreted by PyTorch.

        Parameters:
            root: string
                - Directory containing the image patches
                - "mask" and "nonmask" subdirectories expected
            transform : callable, optional
                - Optional transform to be applied to the sample.
        """
        
        self.root        = root      # Root directory of the image patches
        self.transform   = transform
        prev_dir = os.getcwd()
        
        # File names represent the labels
        os.chdir(self.root)
        # print(self.root)
        self.files = [f for f in glob.glob('*.npy')]
        self.data  = [] # List of tuples of form (data, label) where data is a numpy array
                        # pulled from the loaded label

        for file in self.files:
            label_name = os.path.split(file)[-1][:-4]
            self.data.append((np.load(file, 'r'), label_name))
            
        # Return directory to original working directory
        os.chdir(prev_dir)
        
        
    def __len__(self):
        """Supports array-like length interpretation of the dataset,
        getting the number of image names in the dataset.
        (This does NOT count the labels as entries, only the sub-arrays)
        
        Returns:
            length: integer
        """
        label_sizes = [np.shape(label[0])[0] for label in self.data]
        return(np.sum(label_sizes))
    
    
    def __str__(self):
        """Prints the dataset file path as a string (may be excessively long for large datasets) 
        
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
            label_len = np.shape(self.data[label][0])[0]
            count     = count + label_len
            
        image = torch.from_numpy(self.data[label][0][index - count + label_len])
        label = torch.tensor(label)
        
        if self.transform:
            image = self.transform(image)

        return image, label
    


    def get_labels(self):
        """Obtains the set of label names from the files in the dataset.
        
        Returns:
            labels: [string, string, ...] array
        """
        labels = [label[1] for label in self.data]
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
        for group in self.data:
            if label == group[1]:
                return np.shape(group[0])[0]
        return 0



    def label_to_abs_index(self, index, label):
        """Calculates the absolute index (used by data[i] notation) of an object in the 
        dataset based on the index within the label it is associated in.
        
        Parameters:
            label: string
            index: int
        Returns:
            (image, label): (numpy array, string) tuple
        """
        count     = 0
        label_found = False
        
        for images, class_name in self.data:
            if (label == class_name):
                label_found = True
                break
            else:
                count = count + np.shape(images)[0] - 1
            
        if label_found is False:
            return -1
        else:
            return count + index
