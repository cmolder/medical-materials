import torch
import torch.nn as nn
import torchvision.models as models
from mmac_net.mmac_helpers import DenseLayer, CappedReLU

__all__ = ['MMAC_CNN']

class MMAC_CNN(nn.Module):
    """The MMAC-CNN takes a localized image patche and categorizes its material category
    (K -> sequential layers) and material attribute (M -> auxillary layers) using the A matrix. 
    During training, the A matrix is iteratively updated after each epoch.
    """
    
    def __init__(self, A, psize):
        """Initializes an instance of the MMAC_CNN.

        Parameters:
            A: (k x m) matrix
                The category-attribute matrix
                k: Number of material categories to be predicted
                m: Number of material attributes to be predicted
            psize: int
                The side length of a patch in pixels
        """
        
        super(MMAC_CNN, self).__init__()
        
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Parameter
        # explains why this is not just being assigned as self.A
        A = torch.tensor(A)
        self.register_parameter('A', nn.Parameter(A, requires_grad = False))
        
        self.k = A.shape[0] # Number of material categories
        self.m = A.shape[1] # Number of material attributes
            
        # ResNet34 classifier to k material category classes labelled by humans
        resnet       = models.resnet34(pretrained = True, progress = True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False) # One channel only for greyscale
        resnet       = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*resnet)
        
        # for layer in self.resnet.children():
        #     print('\n---Layer---')
        #     print(type(layer))
        
        relu  = nn.ReLU(inplace = True)
        crelu = CappedReLU(inplace = True)
        
        # i.e. 'brain', 'tumor', 'implant', 'bone', etc.
        # log softmax -> probability of class

        # Add fully connected layers for material category prediction
        self.add_module('fc6',      DenseLayer(512 * (psize // 2**5)**2, 2048, relu))
        self.add_module('fc6_drop', nn.Dropout(p = 0.75))
        self.add_module('fc7',      DenseLayer(2048, 2048, relu))
        self.add_module('fc7_drop', nn.Dropout(p = 0.75))
        self.add_module('logprob',  DenseLayer(2048, self.k, nn.LogSoftmax(dim=-1)))
        
        # Add attribute-constrained auxillary layers
        # for material attribute prediction
        self.add_module('aux1', DenseLayer(16   * (psize // 2**1)**2, self.m, crelu))
        self.add_module('aux2', DenseLayer(16   * (psize // 2**1)**2, self.m, crelu))
        self.add_module('aux3', DenseLayer(128  * (psize // 2**3)**2, self.m, crelu))
        self.add_module('aux4', DenseLayer(256  * (psize // 2**4)**2, self.m, crelu))
        self.add_module('aux5', DenseLayer(512  * (psize // 2**5)**2, self.m, crelu))
        self.add_module('attr_out', DenseLayer(5 * self.m, self.m, crelu))
        
        
    
    def forward(self, x):      
        cache = []
        
        # Pass the value through the Resnet layers first.
        for module_idx, module in enumerate(self.resnet.children()):
            x = module(x) # Pass it through the Resnet layer
                
            # Cache the outputs of the initial pool layer and the
            # sequential blocks because they are inputs to the 
            # auxilliary classifiers for the m attributes
            if isinstance(module, nn.MaxPool2d) or isinstance(module, nn.Sequential):
                cache.append(x)


        x = x.flatten(1, -1)
        
        # Pass the value through the sequential FC layers.
        x = self.fc6(x)
        x = self.fc6_drop(x)

        x = self.fc7(x)      # [2048]
        x = self.fc7_drop(x)
        x = self.logprob(x)  # [k]

        # Predicted material category (i.e k-long vector)
        # is the value of x, the output of ResNet + FC layers
        mat_pred = x
        
        # Take the saved outputs of the pool layers to evaluate the
        # auxillary K material-attribute classifiers
        a1 = self.aux1(cache[0].flatten(1, -1))
        a2 = self.aux2(cache[1].flatten(1, -1))
        a3 = self.aux3(cache[2].flatten(1, -1))
        a4 = self.aux4(cache[3].flatten(1, -1))
        a5 = self.aux5(cache[4].flatten(1, -1))
        
        a_final = self.attr_out(torch.cat([a1, a2, a3, a4, a5], dim=1))
        
        # Return the m material attribute predictions (k_prediction)
        # with intermediate k category predictions (a1, ..., a5)
        return mat_pred, [a1, a2, a3, a4, a5, a_final]
    