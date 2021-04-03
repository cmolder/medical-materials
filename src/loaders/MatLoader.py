import os
import glob
import h5py
from tqdm import tqdm
import numpy as np

from .Loader import Loader

class MatLoader(Loader):
    """Loads .mat files that represent brain tumor scans from the
    following database:
        https://search.datacite.org/works/10.6084/M9.FIGSHARE.1512427.V5
    
    To configure it for another MatLab dataset, edit the image
    and mask paths in the load_all function.
    """
    
    def __init__(self, root, label='image', mask_label='mask', 
                 masked = False, backgrounded = True):
        super().__init__(root, label, mask_label, masked, backgrounded)
        """Loads MAT images.

        Parameters:
            See Loader.py
        """
        


    def load_all(self):
        prev_dir = os.getcwd()
        os.chdir(self.root)
        
        files  = [f for f in glob.glob('*.mat')]
        for i in tqdm(range(len(files)), unit=' files'):
            file = files[i]
            mat_file = h5py.File(file, 'r')
            
            # Normalzie the image
            image = mat_file['/cjdata/image'][()]
            image = image.astype(np.float64)
            image *= 1.0 / image.max()
                
            self.images.append(image)
            self.masks.append(mat_file['/cjdata/tumorMask'][()])

        
        os.chdir(prev_dir)
    
           