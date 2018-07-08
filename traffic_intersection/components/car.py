# Kinematic Car Class
# Tung M. Phan
# California Institute of Technology
# May 3, 2018

import os, sys
sys.path.append("..")
import scipy.io
import numpy as np
from primitives.prim_car import prim_state_dot
from scipy.integrate import odeint
from numpy import cos, sin, tan
main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
primitive_data = main_dir + '/primitives/MA3.mat'

mat = scipy.io.loadmat(primitive_data)
def saturation_filter(u, u_max, u_min):
    """ saturation_filter Helper Function

        the output is equal to u_max if u >= u_max, u_min if u <= u_min, and u otherwise
    """
    return max(min(u, u_max), u_min)


class KinematicCar:
    """Kinematic Car Class

    init_state is [vee, theta, x, y], where vee, theta, x, y are the velocity, orientation, and
    coordinates of the car respectively
    """
    def __init__(self, init_state = [0, 0, 0, 0],
                 L = 50, # length of vehicle
                 a_max = 9.81, # maximum acceleration of vehicle
                 a_min = -9.81, # maximum deceleration of vehicle
                 nu_max = 0.5, # maximum steering input in radians/sec
                 nu_min = -0.5, # minimum steering input in radians/sec)
                 vee_max = 100,
                 color = 'blue', # color of the car)
                 fuel_level = float('inf')): # fuel level of the car - FUTURE FEATURE)
                     if color != 'blue' and color != 'gray':
                         raise Exception("Color must either be blue or gray!")
                     self.init_state = np.array(init_state, dtype="float")
                     self.params = (L, a_max, a_min, nu_max, nu_min, vee_max)
                     self.alive_time = 0
                     self.state = self.init_state
                     self.color = color
                     self.fuel_level = fuel_level

    def state_dot(self,
                  state,
                  t,
                  a,
                  nu):
       """
       state_dot is a function that defines the system dynamics

       Inputs
       state: current state
       t: current time
       a: acceleration input
       nu: steering input

       """
       (L, a_max, a_min, nu_max, nu_min, vee_max) = self.params
       dstate_dt = np.zeros(np.shape(state))
       dstate_dt[0] = saturation_filter(a, a_max, a_min)
       # if already at maximum speed, can't no longer accelerate
       if np.abs(state[1]) >= vee_max and np.sign(a) == np.sign(state[1]):
           dstate_dt[1] = 0
       else:
           dstate_dt[1] = state[0] / L * tan(saturation_filter(nu, nu_max, nu_min))
       dstate_dt[2] = state[0] * cos(state[1])
       dstate_dt[3] = state[0] * sin(state[1])
       return dstate_dt

    def next(self, inputs, dt):
       """
       next is a function that updates the current position of the car when inputs are applied for a duration of dt
       Inputs:
       inputs: acceleration and steering inputs
       dt: integration time

       Outputs:
       None - the states of the car will get updated
       """
       a, nu = inputs

       # take only the real part of the solution
       self.state = odeint(self.state_dot, self.state, t=(0, dt), args=(a, nu))[1]
       # fuel decreases linearly with acceleration/deceleration
       self.fuel_level -= np.abs(a) * dt
       # update alive time
       self.alive_time += dt

    def prim_next(self, prim_id, prim_progress, dt):
       """
       updates with primitive
       Inputs:
       prim_id: primitive ID number
       prim_progress: progress of primitive (a real number between 0 and 1)
       dt: integration time

       Outputs:
       None - the states of the car will get updated
       """
       #TODO: implement compatibility check with primitive to make sure that the params & dynamics match (essentially comparing prim_car and the
       #kinematic model)

       # load primitive data
       prim = mat['MA3'][prim_id,0] # the primitive corresponding to the primitive number
       t_end = prim['t_end'][0,0][0,0] # extract duration of primitive
       N = 5 # hardcoded for now, to be updated


       G_u = np.diag([175, 1.29]) # size of input set
       nu = 2 # number of inputs
       nx = 4 # number of states

       x1 = self.state.reshape((-1,1))
       x2 = prim['x0'][0,0]
       x3 = x_temp0 - prim['x0'][0,0]
       x4 = np.matmul(np.linalg.inv(np.diag([4, 0.02, 4, 4])), (x_temp0-prim['x0'][0,0]))
       x = (np.vstack((x1,x2,x3,x4)))[:,0] # initial state, consisting of actual state and virtual states for the controller

       k = int((prim_progress * t_end) // (t_end / N)) # calculate primitive waypoint

       dist= np.array([[8*(2*np.random.rand())], [0.065*(2*np.random.rand()-1)]]) # random constant disturbance for this time step, disturbance can vary freely. Constant implementation only for easier simulation. TODO: move this outside of this file

       q1 = prim['K'][0,0][k,0].reshape((-1, 1), order='F')
       q2 = 0.5 * (prim['x_ref'][0,0][:,k+1] + prim['x_ref'][0,0][:,k]).reshape(-1,1)
       q3 = prim['u_ref'][0,0][:,k].reshape(-1,1)
       q4 = prim['u_ref'][0,0][:,k].reshape(-1,1)
       q5 = np.matmul(G_u, prim['alpha'][0,0][k*nu:(k+1)*nu]).reshape((-1,1), order='F')
       q = np.vstack((q1,q2,q3,q4,q5)) # parameters for the controller

       self.state = odeint(func = prim_state_dot, y0 = x, t=(0, dt), args=(dist, q))[1,0:4]
       # update alive time
       self.alive_time += dt
       # update progress
       prim_progress += dt / t_end
       return prim_progress

# TESTING
prim_id = 0
prim = mat['MA3'][prim_id,0] # the primitive corresponding to the primitive number
x0 = np.array(prim['x0'][0,0]) + np.matmul(np.diag([4, 0.02, 4, 4]),2*np.random.rand(4,1)-np.ones((4,1))) # random state in initial set TODO: incorporate self.state
my_car = KinematicCar(init_state = np.reshape(x0, (-1, 1)))
progress = 0
while progress <= 1:
    progress = my_car.prim_next(prim_id = 0, prim_progress = progress, dt = 0.1)
