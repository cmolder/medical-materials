from torch import nn
import torch
import torch.nn.functional as F
import torchvision.models as models




class D_CNN(nn.Module):
    """Siamese NN that learns material similarity to generate 
    a D matrix.
    """
    
    def __init__(self):
        
        super().__init__()
        resnet       = models.resnet34(pretrained = False, progress = True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False) # One channel only for greyscale
        resnet       = list(resnet.children())[:-2]
        
        self.resnet  = nn.Sequential(*resnet)
        
        # Our linear layers
        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, 2)
    
    def forward(self, data):

        # Split the batches, which are grouped into (n+1) sets
        # where (n+1) is the number of images in a group
        # and n is the number of classes
        #
        # Map [bth_sz, n + 1, img_sz, img_sz] image tensor to
        # a (n + 1) long list of [bth_sz, 1, img_sz, img_sz] tensors
        #
        # Results is a (n + 1) long list of [bth_sz * # linear output] tensors.
        data = torch.split(data, 1, dim=1)
        
        results = []
            
        for i in range(len(data)):
            x = data[i]
            x = self.resnet(x)
            
            x = x.view(x.shape[0], -1)
            x = self.linear1(x)
            results.append(F.relu(x))
        
        # Result is the differences of reference vs comp for each
        # comparison image
        #
        # We work backwards through the results to avoid altering
        # the reference image results before altering the other images
        for i in range(len(results) - 1, -1, -1):
            results[i] = torch.abs(results[i] - results[0])


        # Stack the results into a [n + 1, bth_sz, 2] tensor
        # Each n is the guess for image n of the reference images
        #
        # If the value is higher for dim3 index 0, then the NN thinks they are of the same class.
        # If the value is higher for dim3 index 1, then the NN thinks they are of a different class.
        results = [self.linear2(res) for res in results]
        results = torch.stack(results)
        
        # Resize the results into a [bth_sz, n + 1, 2] tensor
        results = results.transpose(0, 1)
        return results
            