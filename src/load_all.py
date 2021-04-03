import sys

import matplotlib.pyplot as plt
import numpy as np

from loaders import MatLoader, DcmLoader

def main(data_path):
    """Loads data from 'Brain-Tumor-Progression', 'Brain Tumor Dataset', 
    and 'CHECK' datasets and generates relevant patches.

    Parameters:
        data_root: string
        - Root path of the datasets
        - Directory structure:
            - 'Brain Tumor Dataset' should be in 'data_root/bt-data'
            - 'Brain-Tumor-Progression' should be in 'data_root/tcia-data'
            - 'CHECK' should be in 'data_root/check-data'
            - Can add or remove datasets as desired by editing the file;
              Implementations can handle Matlab (.mat) and DCM (.dcm) images
        - Patch output:
            - data_root/patch-set/
    """

    
    # Load brain tumor scans from brain-tumor set (.mat)
    bt = MatLoader(root = f'{data_path}/bt-data', label='brain', mask_label='tumor', masked=True)
    bt.gen_patches(size=32, tolerance=0.1, samples=30000, cutoff=10000000, min_avg_val=0.25) 
    bt.save_patches(ftype='*', path = f'{data_path}/patch-set/', val=0.2, test=0.2)
    
    # for i in range(2524,2525):
    #     bt.display(i)
    #     bt.display(i, mask_overlay = True)
    del bt
    
    # Load brain tumor scans from TCIA set (.dcm)
    tcia = DcmLoader(root = f'{data_path}/tcia-data', label='brain', mask_label='tumor', masked=True)
    tcia.gen_patches(size=32, tolerance=0.1, cutoff=4000000, min_avg_val=0.25)  
    tcia.save_patches('*', path = f'{data_path}/patch-set/', val=0.2, test=0.2)
    del tcia

    # Load knee x-ray scans from CHECK set (.dcm)
    check = DcmLoader(f'{data_path}/check-data/test/', label='bone', backgrounded=False)
    check.gen_patches(size = 32, samples = 13525, min_avg_val = 0.25, max_avg_val = 0.85)
    check.save_patches('*', path=f'{data_path}/patch-set/', val=0.2, test=0.2)
    
    # for i in range(115,116):
    #    check.display(i)
    del check
    
    
    # Probe what our loaders have loaded
    # result = np.load(f'{DATA_PATH}/patch-set/npy/train/brain.npy', 'r')
    # print(np.shape(result))
    

if __name__ == '__main__':
    # Launch program using python load_all.py <data_root>
    try:
        dp = sys.argv[1]
    except:
        print('Error retreiving data root. Run the script as "python test.py <data_root>".')
        quit()

    main(dp)
