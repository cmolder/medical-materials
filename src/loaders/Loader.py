import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import imageio

from tqdm import tqdm
from random import randrange

# Global variable to keep track of the title
# of the background class, this should not
# be overridden by anything to keep it
# the same for all
background_label = 'background'

class Loader():
    def __init__(self, root, label='image', mask_label='mask', 
                 masked = False, backgrounded = True):
        """Parent class for loading medical images.

        Parameters:
            root: string
            - Root directory of the images (required) / masks (optional)
            label: string
            - What to name the desired material category (out-of-mask category)
            mask_label: string
            - What to name the desired in-mask material category (in-mask category)
            masked: boolean
            - If true, loads each image's mask and parses it to generate both in-mask,
              and out-of-mask material patches.
            backgrounded: boolean
            - If true, assumes dark regions of the image are the background, and separates
              them from the out-of-mask category.

        """
        self.root = root # Root directory of the images / masks

        self.masked       = masked
        self.backgrounded = backgrounded
        
        self.label      = label
        self.mask_label = mask_label

        self.images = []
        self.masks  = [] # Make sure images are normalized to [0,1] on loading.
        
        self.image_patches = []
        self.mask_patches  = []
        self.bkgd_patches  = [] # Background category patches
        
        self.load_all()
    
    def load_all(self):
        """Function to be implemented by a child class,
        loads the images/masks/etc. into a standard format
        for patch generation.
        """
        pass
    
    def display(self, index, mask_overlay = False, save = False):
        """Takes the image layer given by (image)
        and displays a plot of the image layer with the mask overlayed on the image
        """
        
        image_data = self.images[index]
        
        plt.figure(index)
        plt.title(f'Image {index}')
        plt.imshow(image_data, cmap='bone')
        
        if save:
            plt.savefig(f'{index}img.png', dpi = 400)
        
        if self.masked and mask_overlay:
            mask_data  = self.masks[index]
            plt.imshow(mask_data, cmap='viridis', alpha=0.4)
            
            if save:
                plt.savefig(f'{index}mask.png', dpi = 400)
        
        plt.show()
        
    def __getitem__(self, index):
        """Returns the raw numpy array that represents the image at
        the given index.
        """
        if self.masked:
            return self.images[index], self.masks[index]
        else:
            return self.images[index], None
            # return self.images[index] # If the above gives you fits
        

    def gen_patches(self, size = 48, samples = 10000, cutoff = 1000000, 
                    tolerance = 0.1, min_avg_val = 0.1, max_avg_val = 1.0,
                    max_bkgd_avg = 0.05, overlap = 16, balanced = True):
        """Selects random patches from the image, compares the location to the mask 
        to see how much mask is in the image, labels it accordingly, and saves the 
        patch coordinates if it highly correaltes to inside / outside the mask.
        
        Parameters:
            size: int
                - Side length of the patch in pixels, patch is n x n pixels square
            samples: int
                - Number of patches that we maximally desire from the patch
                generation sequence.
            tolerance: float
                - The maximum ratio of the image that is NOT the associated label
                - (i.e a tolerance of 0.3 means images that are 30% tumor and 70% 
                healthy would be classified as healthy, but an image that is 
                40% tumor and 60% healthy would be ignored.)
            cutoff: int
                - The maximum number of samples that will be look at before stopping
                the patch generation sequence.
            min_avg_val: float
                - The minimum average brightness of the image. This is to avoid
                certain patches of scans i.e. the background from being loaded
                into the dataset.
                - Range from [0, 1]
            overlap: int
                - The minumum pixel overlap between similar patches, i.e.
                the interval between similar patches.
            balanced: bool
                - If true, the loader stores an equal amount of mask and non-mask 
                patches, regardless of the number of valid patches found
                (given that masked is True for the loader)
            max_bkgd_avg: float
                - The maximum average brightness of an image that is deemed
                to be background.
                - Range from [0, 1]
        """
        
        num_mask  = 0 # number of mask patches saved
        num_image = 0 # number of non-mask patches saved
        num_bkgd  = 0 # number of background patches saved
        
        if not self.masked:
            balanced = False
        
        
        for i in tqdm(range(cutoff), unit=' samples'):
                
            # If we have reached the number of samples requested, quit
            if num_mask + num_image > samples:
                break
            
            # Select a random image in the loaded dataset.
            rand_image = randrange(0, len(self.images))
         
            # Get the actual image data
            # (If masked is True, assuming every image has matching masks)
            layer_image = self.images[rand_image]
            
            if self.masked:
                layer_mask  = self.masks[rand_image]
            
            # Select a random x and y inside the image that
            # will be the top-left corner of this patch
            # (x, y are multiples of the overlap)
            rand_x = randrange(0, int((len(layer_image)- size)/overlap)) * overlap
            rand_y = randrange(0, int((len(layer_image[0])- size)/overlap)) * overlap
            
            cropped_image = layer_image[rand_x:rand_x+size, rand_y:rand_y+size]
            average_val  = np.average(cropped_image)
            
            # Look at the cropped patch of the mask to see if it falls within the tolerance.
            if self.masked:
                cropped_mask = layer_mask[rand_x:rand_x+size, rand_y:rand_y+size]
                in_mask_pct  = np.count_nonzero(cropped_mask) * 1.0 / np.size(cropped_mask)
           
            
            # If the in_mask_pct is > (1 - tolerance) or < tolerance
            # AND the image isn't heavily black background (avg val < min_avg_val) then it is okay.
            # i.e. > 70% in mask or < 30% in mask if the tolerance is 0.3.
            if(average_val > min_avg_val and average_val < max_avg_val):
                
                # If we have masks, then check if the majority is in the mask -> to mask patches
                if(self.masked and in_mask_pct > (1 - tolerance)): # (majority in mask)
                    if(not balanced or num_mask <= num_image):
                        new_patch = (rand_image, rand_x, rand_y, size)
                        
                        if(new_patch not in self.mask_patches):
                            self.mask_patches.append(new_patch)
                            num_mask += 1             

                # If we have masks, then check if majority is not in mask -> to image patches
                elif(self.masked and in_mask_pct < tolerance):     # (majority non-mask)
                    if(not balanced or num_image <= num_mask):
                        new_patch = (rand_image, rand_x, rand_y, size)
                        
                        if(new_patch not in self.image_patches):
                            self.image_patches.append(new_patch)
                            num_image +=  1
                            
                # If we don't have masks, just add the patch given it hits the min/max vals.
                elif(not self.masked):
                    new_patch = (rand_image, rand_x, rand_y, size)
                    if(new_patch not in self.image_patches):
                            self.image_patches.append(new_patch)
                            num_image += 1
               
            # If the image is heavily black background (avg val < max_bkgd_avg), then
            # we should save it as a background-class patch.
            elif (average_val <= max_bkgd_avg):
                if(not balanced or num_bkgd <= num_image):
                    new_patch = (rand_image, rand_x, rand_y, size)
                    
                    if(new_patch not in self.bkgd_patches):
                        self.bkgd_patches.append(new_patch)
                        num_bkgd += 1
            
            
        print(f'# of mask patches      : {len(self.mask_patches)}')
        print(f'# of image patches     : {len(self.image_patches)}')
        print(f'# of background patches: {len(self.bkgd_patches)}')
        
    
            
    def save_patches(self, ftype, path='', val=0.0, test=0.0, seed = np.random.randint(100000)):
        """Takes the image_patches and mask_patches lists and saves the combined
        arrays of the mask/image patches to disk using one of the write_patches_*
        functions, with the functionality of splitting the lists into validation,
        testing, and training sets.
        
        The image structure is a Numpy array.
        - Axis 0 is the list of images
        - Axis 1 is the x-coordinate of a particular image
        - Axis 2 is the y-coordiante of a particular image
        """
        
        # Split the possible indexes for images into their proportional
        # validation, testing, and training set sizes
        val_split  = int(np.floor(len(self.image_patches) * val))
        test_split = int(np.floor(len(self.image_patches) * test)) + val_split
        indexes    = list(range(len(self.image_patches)))
        
        # print(f'last val  before {val_split}')
        # print(f'last test before {test_split}')
        
        if self.masked:
            valm_split  = int(np.floor(len(self.mask_patches) * val))
            testm_split = int(np.floor(len(self.mask_patches) * test)) + valm_split
            m_indexes   = list(range(len(self.mask_patches)))
            
        if self.backgrounded:
            valb_split  = int(np.floor(len(self.bkgd_patches) * val))
            testb_split = int(np.floor(len(self.bkgd_patches) * test)) + valb_split
            b_indexes   = list(range(len(self.bkgd_patches)))
    
    
        # Shuffle the splits
        np.random.seed(seed)
        np.random.shuffle(indexes)
        vals, tests, trains = indexes[:val_split], indexes[val_split:test_split], indexes[test_split:]
        
        if self.masked:
            np.random.shuffle(m_indexes)
            valsm, testsm, trainsm = m_indexes[:valm_split], m_indexes[valm_split:testm_split], m_indexes[testm_split:]
            
        if self.backgrounded:
            np.random.shuffle(b_indexes)
            valsb, testsb, trainsb = b_indexes[:valb_split], b_indexes[valb_split:testb_split], b_indexes[testb_split:]

        
        # Splice the images of the patches from the
        # larger medical scans.
        val_patches   = []
        test_patches  = []
        train_patches = []
        
        for count, i in enumerate(itertools.chain(vals, tests, trains)):
            data  = self.image_patches[i]
            loc   = data[0] # Location of image in self.images array
            x     = data[1] # x coordinate of top-left corner of patch in the image
            y     = data[2] # y coordinate
            size  = data[3] # one-length side of the patch
        
            patch = np.copy(self.images[loc][x:x+size, y:y+size])
            patch = np.array(patch).astype(np.float32)

            # Train sample
            if count >= val_split and count >= test_split: 
                # if i in trains:
                #     print(f'train {count} {i}')
                # else:
                #     print('error, not a train!')
                train_patches.append(patch)
            # Test sample
            elif count >= val_split:                    
                # if i in tests:
                #     print(f'test  {count} {i}')
                # else:
                #     print("error, not a test!")
                test_patches.append(patch)
            # Validation sample
            else: 
                # if i in vals:
                #     print(f'val   {count} {i}')
                # else:
                #     print("error, not a val!")  
                val_patches.append(patch)
        
        # NOTE: This does not save patches of the MASK, but rather patches
        # of the image that fall within the mask boundaries. Very important
        # distinction to be made! 
        if self.masked:
            valm_patches   = []
            testm_patches  = []
            trainm_patches = []
            
            for count, i in enumerate(itertools.chain(valsm, testsm, trainsm)):
                data  = self.mask_patches[i]
                loc   = data[0] # Location of image in self.images array
                x     = data[1] # x coordinate of top-left corner of patch in the image
                y     = data[2] # y coordinate
                size  = data[3] # one-length side of the patch
            
                patch = np.copy(self.images[loc][x:x+size, y:y+size]) # Save image of location, not mask
                patch = np.array(patch).astype(np.float32)
    
                if count >= valm_split and count >= testm_split: # Train sample
                    trainm_patches.append(patch)
                elif count >= valm_split:                        # Test sample         
                    testm_patches.append(patch)
                else:                                            # Validation sample
                    valm_patches.append(patch)
                    
        if self.backgrounded:
            valb_patches   = []
            testb_patches  = []
            trainb_patches = []
            
            for count, i in enumerate(itertools.chain(valsb, testsb, trainsb)):
                data  = self.bkgd_patches[i]
                loc   = data[0] # Location of image in self.images array
                x     = data[1] # x coordinate of top-left corner of patch in the image
                y     = data[2] # y coordinate
                size  = data[3] # one-length side of the patch
            
                patch = np.copy(self.images[loc][x:x+size, y:y+size]) # Save image of location
                patch = np.array(patch).astype(np.float32)
    
                if count >= valb_split and count >= testb_split:  # Train sample
                    trainb_patches.append(patch)
                elif count >= valb_split:                         # Test sample                       
                    testb_patches.append(patch)
                else:                                             # Validation sample
                    valb_patches.append(patch)
        
        # Save mask patches
        if self.masked:
            if ftype == 'png' or ftype == '*':
                self.write_patches_png([valm_patches, testm_patches, trainm_patches], path, self.mask_label)
            if ftype == 'npy' or ftype == '*':
                self.write_patches_npy([valm_patches, testm_patches, trainm_patches], path, self.mask_label)
                
        # Save background patches
        if self.backgrounded:
            if ftype == 'png' or ftype == '*':
                self.write_patches_png([valb_patches, testb_patches, trainb_patches], path, background_label)
            if ftype == 'npy' or ftype == '*':
                self.write_patches_npy([valb_patches, testb_patches, trainb_patches], path, background_label)
                
        # Save non-mask patches
        if ftype == 'png' or ftype == '*':
            self.write_patches_png([val_patches, test_patches, train_patches], path, self.label)
        if ftype == 'npy' or ftype == '*':
            self.write_patches_npy([val_patches, test_patches, train_patches], path, self.label)
        

   
    def write_patches_npy(self, patches, path, label):
        """Writes the patch arrays to disk as *.npy files. This is only done for one label
        at a time (i.e., you must pass in mask, image, and background patches separately
        with the adequate paths.)
        
        Notes:
            The npy-stored images are normalized to the range [0, 1].
        """
        # Patches are in form [validation patches, testing patches, training patches]
        # Some can be empty and that is OK.
        val_patches, test_patches, train_patches = patches
        
        val_data_path = os.path.join(path, 'npy', 'val', f'{label}.npy')
        val_data      = np.load(val_data_path) if os.path.exists(val_data_path) else None
        
        test_data_path = os.path.join(path, 'npy', 'test', f'{label}.npy')
        test_data  = np.load(test_data_path) if os.path.exists(test_data_path) else None
        
        train_data_path = os.path.join(path, 'npy', 'train', f'{label}.npy')
        train_data = np.load(train_data_path) if os.path.exists(train_data_path) else None
        
        if val_data is not None and len(val_patches) > 0:
            val_patches = np.append(val_data, val_patches, axis=0)
        np.save(val_data_path, val_patches)

        if test_data is not None and len(test_patches) > 0:
            test_patches = np.append(test_data, test_patches, axis=0)
        np.save(test_data_path, test_patches)

        if train_data is not None and len(train_patches) > 0:
            train_patches = np.append(train_data, train_patches, axis=0)
        np.save(train_data_path, train_patches)
       
            

    def write_patches_png(self, patches, path, label):
        """Writes the patch arrays to disk as *.png files. This is only done for one label
        at a time (i.e., you must pass in mask, image, and background patches separately
                with the adequate paths.)
        
        Notes:
            The png-stored images are normalized to the range [0, 255] as integer values.
        """
        # Patches are in form [validation patches, testing patches, training patches]
        # Some can be empty and that is OK.
        val_patches, test_patches, train_patches = patches
        
        val_data_f   = os.path.join(path, 'png', 'val', f'{label}')   # Data folders
        test_data_f  = os.path.join(path, 'png', 'test', f'{label}')
        train_data_f = os.path.join(path, 'png', 'train', f'{label}')
        
        for count, pth in enumerate(val_patches):
            imageio.imwrite(os.path.join(val_data_f, f'{count}.png'), (pth * 255.0).astype(np.uint8))
        for count, pth in enumerate(test_patches):
            imageio.imwrite(os.path.join(test_data_f, f'{count}.png'), (pth * 255.0).astype(np.uint8))
        for count, pth in enumerate(train_patches):
            imageio.imwrite(os.path.join(train_data_f, f'{count}.png'), (pth * 255.0).astype(np.uint8))
        