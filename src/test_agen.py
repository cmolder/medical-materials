import numpy as np
from d_net import create_A

"""Tests A matrix generation
"""
D = np.load('d.npy')
m = 4

A = create_A(D, m)

print('Saving A to a.npy...')
np.save('a.npy', A)