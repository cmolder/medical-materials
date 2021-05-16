import torch
import torch.nn as nn
import torchvision.models as models
from mmac_net.mmac_helpers import DenseLayer, CappedReLU

__all__ = ['MMAC_CNN_VGG16']

class MMAC_CNN_VGG16(nn.Module):
    """The MMAC-CNN takes a localized image patche and categorizes its material category
    (K -> sequential layers) and material attribute (M -> auxillary layers) using the A matrix. 
    During training, the A matrix is iteratively updated after each epoch.
    
    This version of the MMAC-CNN uses a VGG-16 backbone, like (Schwartz and Nishino 2020), for
    comparison to our upgraded version with a ResNet34 backbone.
    """
    
    def __init__(self, A, psize):
        """Initializes an instance of the MMAC_CNN (VGG16).

        Parameters:
            A: (k x m) matrix
                The category-attribute matrix
                k: Number of material categories to be predicted
                m: Number of material attributes to be predicted
            psize: int
                The side length of a patch in pixels
        """
        
        super(MMAC_CNN_VGG16, self).__init__()
        
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Parameter
        # explains why this is not just being assigned as self.A
        A = torch.tensor(A)
        self.register_parameter('A', nn.Parameter(A, requires_grad = False))
        
        self.k = A.shape[0] # Number of material categories
        self.m = A.shape[1] # Number of material attributes

            
        # VGG16 classifier to k material category classes labelled by humans
        vgg   = models.vgg16(pretrained = True, progress = True)
        vgg   = list(vgg.children())[:-2]
        
        # Reconfigure the first conv layer to use one channel only,
        # as the input images are greyscale.
        vgg[0][0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.vgg = nn.Sequential(*vgg)
        
        #for layer in self.vgg.children():
        #    print('\n---Layer---')
        #    print(type(layer))
        #    print(layer)

        relu  = nn.ReLU(inplace = True)
        crelu = CappedReLU(inplace = True)
        
        # i.e. 'brain', 'tumor', 'implant', 'bone', etc.
        # log softmax -> probability of class

        # Replace pretrained fully connected layers for ours, to allow
        # for material category prediction
        self.add_module('fc6',      DenseLayer(512 * (psize // 2**5)**2, 2048, relu))
        self.add_module('fc6_drop', nn.Dropout(p = 0.75))
        self.add_module('fc7',      DenseLayer(2048, 2048, relu))
        self.add_module('fc7_drop', nn.Dropout(p = 0.75))
        self.add_module('logprob',  DenseLayer(2048, self.k, nn.LogSoftmax(dim=-1)))
        
        # Add attribute-constrained auxillary layers
        # for material attribute prediction
        self.add_module('aux1', DenseLayer(64   * (psize // 2**1)**2, self.m, crelu))
        self.add_module('aux2', DenseLayer(128  * (psize // 2**2)**2, self.m, crelu))
        self.add_module('aux3', DenseLayer(256  * (psize // 2**3)**2, self.m, crelu))
        self.add_module('aux4', DenseLayer(512  * (psize // 2**4)**2, self.m, crelu))
        self.add_module('aux5', DenseLayer(512  * (psize // 2**5)**2, self.m, crelu))
        self.add_module('attr_out', DenseLayer(5 * self.m, self.m, crelu))
        
        #for layer in self.vgg.children():
        #    print('\n---Layer---')
        #    print(layer)
    
    def forward(self, x):      
        cache = []
        
        #
        # First, pass the value through the headless VGG16 layers.
        #
        for module_idx, module in enumerate(self.vgg.children()):
            
            # If we are going through one of the VGG's sequential blocks,
            # pass the value through each submodule independently, so we
            # can identify the MaxPool2d layers and cache the auxiliary data.
            #print('\n-----Architecture-----')
            if isinstance(module, nn.Sequential):
                for submodule_idx, submodule in enumerate(module.children()):
                    #print(submodule)
                    #print('\n', submodule)
                    x = submodule(x)
                    
                    #print(x[0,:,0,0])
                    
                    # Cache the outputs of the initial pool layer and the
                    # sequential blocks because they are inputs to the 
                    # auxilliary classifiers for the m attributes
                    if isinstance(submodule, nn.MaxPool2d):
                        cache.append(x)
            else:
                #print(module)
                x = module(x) # Pass it through the module as a single unit
                if isinstance(module, nn.MaxPool2d):
                    cache.append(x)

        #
        # Second, pass the value through the sequential FC layers.
        #
        x = x.flatten(1, -1)
        x = self.fc6(x)
        x = self.fc6_drop(x)

        x = self.fc7(x)      # [2048]
        x = self.fc7_drop(x)
        x = self.logprob(x)  # [k]

        # Predicted material category (i.e k-long vector)
        # is the value of x, the output of ResNet + FC layers
        mat_pred = x
        
        #
        # Finally, use the cached outputs of the MaxPool2d layers to eavluate
        # the auxillary material attribute classifiers
        #
        a1 = self.aux1(cache[0].flatten(1, -1))
        a2 = self.aux2(cache[1].flatten(1, -1))
        a3 = self.aux3(cache[2].flatten(1, -1))
        a4 = self.aux4(cache[3].flatten(1, -1))
        a5 = self.aux5(cache[4].flatten(1, -1))
        
        a_final = self.attr_out(torch.cat([a1, a2, a3, a4, a5], dim=1))
        
        # Return the m material attribute predictions (k_prediction)
        # with intermediate k category predictions (a1, ..., a5)
        return mat_pred, [a1, a2, a3, a4, a5, a_final]
    