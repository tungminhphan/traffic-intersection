# Primitive Controller
# Authors: Bastian Schürmann, Tùng M. Phan
# California Institute of Technology
# July 4, 2018

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import controlled_car as dynamics_model
import pdb
mat = scipy.io.loadmat('MA3.mat')

prim_num = 0 # primitive number
prim = mat['MA3'][prim_num,0] # the primitive corresponding to the primitive number

if not prim['controller_found'][0,0][0,0]:
    print('Maneuver does not exist yet. Please choose different maneuver_number.')

N = 5 # number of segments for the reference trajectory
G_u = np.diag([175, 1.29]) # size of input set
nu = 2 # number of inputs
nx = 4 # number of states

x = np.zeros([4*nx, N+1])
x_temp0 = np.array(prim['x0'][0,0]) + np.matmul(np.diag([4, 0.02, 4, 4]),2*np.random.rand(nx,1)-np.ones((nx,1))) # random state in initial set

x_temp0 = np.array([[1.5586],[-0.0073],[3.6018],[321.2756]]) #TODO: REMOVE

x1 = x_temp0
x2 = prim['x0'][0,0]
x3 = x_temp0 - prim['x0'][0,0]
x4 = np.matmul(np.linalg.inv(np.diag([4, 0.02, 4, 4])), (x_temp0-prim['x0'][0,0]))
x[:,0] = (np.vstack((x1,x2,x3,x4)))[:,0]
for k in range(0,N):
    dist= np.array([[8*(2*np.random.rand())], [0.065*(2*np.random.rand()-1)]]) # random constant disturbance for this time step, disturbance can vary freely. Constant implementation only for easier simulation.
    dist= np.array([[0.1*(k+1)], [-0.1*(k+1)]]) # TODO: REMOVE
    q1 = prim['K'][0,0][k,0].reshape((-1, 1), order='F')
    q2 = 0.5 * (prim['x_ref'][0,0][:,k+1] + prim['x_ref'][0,0][:,k]).reshape(-1,1)
    q3 = prim['u_ref'][0,0][:,k].reshape(-1,1)
    q4 = prim['u_ref'][0,0][:,k].reshape(-1,1)
    q5 = np.matmul(G_u, prim['alpha'][0,0][k*nu:(k+1)*nu,:]).reshape((-1,1), order='F')
    q = np.vstack((q1,q2,q3,q4,q5)) # parameters for the controlller
    t_end =prim['t_end'][0,0][0,0]
    x_temp = odeint(dynamics_model.controlled_car, x[:,k], np.linspace(0, t_end/5.,65), args=(dist, q))
    x[:,k+1] = x_temp[-1,:]

# plot of the resulting trajectory in different dimensions over time. Comparison with reference trajectory in red.    
x_ref = prim['x_ref'][0,0]

plt.subplot(221)
plt.plot(x[0,:])
plt.plot(x_ref[0,:])
plt.legend(('real', 'ref'))
plt.xlabel('t')
plt.ylabel('v')
plt.subplot(222)
plt.plot(x[1,:])
plt.plot(x_ref[1,:])
plt.legend(('real', 'ref'))
plt.xlabel('t')
plt.ylabel('psi')
plt.subplot(223)
plt.plot(x[2,:])
plt.plot(x_ref[2,:])
plt.legend(('real', 'ref'))
plt.xlabel('t')
plt.ylabel('x')
plt.subplot(224)
plt.plot(x[3,:])
plt.plot(x_ref[3,:])
plt.xlabel('t')
plt.ylabel('y')
plt.legend(('real', 'ref'))
plt.show()
