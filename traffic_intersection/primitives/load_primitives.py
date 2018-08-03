# Load Primitives
# Tung M. Phan
# July 23rd, 2018
# California Institute of Technology
import os
import scipy.io
dir_path = os.path.dirname(os.path.realpath(__file__))
primitive_data = dir_path + '/MA3.mat'
mat = scipy.io.loadmat(primitive_data)
num_of_prims = mat['MA3'].shape[0]

def get_prim_data(prim_id, data_field):
    '''
    This function simplifies the process of extracting data from the .mat file
    Input:
    prim_id: the index of the primitive the data of which we would like to return
    data_field: name of the data field (e.g., x0, x_f, controller_found etc.)

    Output: the requested data
    '''
    return mat['MA3'][prim_id,0][data_field][0,0][:,0]
