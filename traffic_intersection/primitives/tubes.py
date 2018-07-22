#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Tube Computation for Primitives
# Author: Tùng M. Phan
# California Institute of Technology
# July 20, 2018

import os, sys
sys.path.append("..")
import numpy as np
import prepare.collision_check

def tubes_overlapped(tube_1, tube_2):
    return tube_1 == tube_2

def primitive_to_tube(primitive_id):
    pass

def compute_collision_dictionary(primitive_set):
    col_dict = dict()
    for primitive in primitive_set:
        col_dict[primitive]

#########################################################################################
#                                                                                       #
#                                   TESTING                                             #
#                                                                                       #
#########################################################################################
def get_prim_data(prim_id, data_field):
    '''
    This function simplifies the process of extracting data from the .mat file
    Input:
    prim_id: the index of the primitive the data of which we would like to return
    data_field: name of the data field (e.g., x0, x_f, controller_found etc.)

    Output: the requested data
    '''
    return mat['MA3'][prim_id,0][data_field][0,0][:,0]

import os
import scipy.io
# set dir_path to current directory
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

car_scale_factor = 0.01 # scale for when L = 50
pedestrian_scale_factor = 0.32
# load primitive data
primitive_data_path = dir_path + '/primitives/MA3.mat'
mat = scipy.io.loadmat(primitive_data_path)
num_of_prims = mat['MA3'].shape[0]

a = mat['MA3'][0,0]['x_ref'][0,0]
b = get_prim_data(0, 'x0')
c = get_prim_data(0, 'x_f')

print(a[:,0])
print(a[:,1])
print(a[:,2])
print(a[:,3])
print(a[:,4])
print(a[:,5])
print(b)
print(c)

import matplotlib.pyplot as plt

x = a[2,:]
y = a[3,:]
plt.plot(x,y)
plt.show()
