# Primitive Controlled Car Dynamics
# Authors: Bastian Schürmann, Tùng M. Phan
# California Institute of Technology
# July 4, 2018
import numpy as np

def controlled_car(t, x, u, q):
    x_real = x[0:4] # actual state
    x_ref_nl = x[4:8] # reference state for the center
    x_ref_lin = x[8:12] # reference state for linearized dynamics
    beta = x[12:16] # parameters for intial state

    w = u[0:2] # disturbance
    K = q[0:8].reshape((2,4), order='F')
    x_lin = q[8:12] # linearization point states
    u_lin = q[12:14] # linearization point inputs
    u_ff_nl = q[14:16] # feedforward input for refence trajectory
    u_ff_lin_G = q[16:24].reshape((2,4), order='F') # feedforward input for generators
    u_ff_lin = u_ff_lin_G*beta # feedforward input based on actual initial state

    u_fb=-np.matmul(K, x_real-x_rel_nl-x_ref_lin) # feedback input

    u_ff = u_ff_nl + u_ff_lin # combined feedfoward input

    # linearized dynamics
    A = np.array([ ...
                 [0,0,0,0], ...
                 [u_lin[1]/50.,0,0,0], ...
                 [np.cos(x_lin[1]),-x_lin[0]*np.sin(x_lin[1]),0,0], ...
                 [np.sin(x_lin[1]),x_lin[0]*np.cos(x_lin[1]),0,0]])

    B = np.array([ ...
                [1,0], ...
                [0, x_lin[0]/50.], ...
                [0, 0], ...
                [0, 0]])

    f[0:4,1] = np.vstack((u_ff[0]+u_fb[0]+w[0],x[0]/50.*(u_ff[1]+u_fbb[1]+w[1]), np.cos(x[1])*x[0], np.sin(x[1]*x[0]))) # actual dynamics
    f[4:8,1] = np.vstack((u_ff_nl[0], x[4]/50. * (u_ff_nl[1]), np.cos(x[5]) * x[4], sin(x[5])*x[4] )) # reference dynamics for center
    f[8:12,1] = np.matmul(A,x_ref_lin) + np.matmul(B,u_ff_lin) # linearized reference dynamics
    f = np.vstack((f, np.zeros((4,1))))
    return f


