# Disturbance Assumption of Primitives
# Tung M. Phan
# California Institute of Technology
# July 17th, 2018
import numpy as np


def get_disturbance():
    '''
    Returns random disturbance
    '''
    return np.array([[8*(2*np.random.rand())], [0.065*(2*np.random.rand()-1)]]) # a random constant disturbance for our primitives
