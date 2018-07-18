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
from prepare.queue import Queue
from PIL import Image
from assumes.disturbance import get_disturbance


dir_path = os.path.dirname(os.path.realpath(__file__))
blue_car_fig = dir_path + "/imglib/cars/blue_car.png"
gray_car_fig = dir_path + "/imglib/cars/gray_car.png"
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
                 vee_max = 100, # maximum velocity
                 color = 'blue', # color of the car
                 prim_queue = None, # queue of primitives, each item in the queue has the form (prim_id, prim_progress) where prim_id is the primitive ID and prim_progress is the progress of the primitive)
                 fuel_level = float('inf')): # TODO: fuel level of the car - FUTURE FEATURE)
                     if color != 'blue' and color != 'gray':
                         raise Exception("Color must either be blue or gray!")
                     self.params = (L, a_max, a_min, nu_max, nu_min, vee_max)
                     self.alive_time = 0
                     self.state = np.array(init_state, dtype='float')
                     self.color = color
                     self.extended_state = None # extended state required for Bastian's primitive computation
                     if prim_queue == None:
                         self.prim_queue = Queue()
                     else:
                         self.prim_queue = prim_queue
                     self.fuel_level = fuel_level
                     if color == 'blue':
                         self.fig = Image.open(blue_car_fig)
                     else:
                         self.fig = Image.open(gray_car_fig)

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

       # TODO: temporary fix to floating problem
       if a == 0:
           self.state[0] = np.sign(self.state[0]) * abs(self.state[0]) * dt * 0.05

    def extract_primitive(self):
       #TODO: rewrite the comment below
       """
       This function updates the primitive queue and picks the next primitive to be applied. When there is no more primitive in the queue, it will
       return False

       """
       while self.prim_queue.len() > 0:
           if self.prim_queue.top()[1] < 1: # if the top primitive hasn't been exhausted
               prim_id, prim_progress = self.prim_queue.top() # extract it
               return prim_id, prim_progress
           else:
               self.prim_queue.pop() # pop it
       return False

    def prim_next(self, dt):
       """
       updates with primitive, if no primitive available, update with next with zero inputs
       Inputs:
       dt: integration time

       Outputs:
       None - the states of the car will get updated
       """
       #TODO: implement compatibility check with primitive to make sure that the params & dynamics match (essentially comparing prim_car and the
       #kinematic model)
       if self.extract_primitive() == False: # if there is no primitive to use
           self.next((0, 0), dt)
       else:
           prim_id, prim_progress = self.extract_primitive()
           # load primitive data TODO: make this portion of the code more automated
           prim = mat['MA3'][prim_id,0] # the primitive corresponding to the primitive number
           t_end = prim['t_end'][0,0][0,0] # extract duration of primitive
           N = prim['K'][0,0].shape[0] # number of subintervals encoded in primitive
           G_u = np.diag([175, 1.29]) # this diagonal matrix encodes the size of input set (a constraint)
           nu = 2 # number of inputs
           nx = 4 # number of states

           if prim_progress == 0: # compute initial extended state
               x1 = self.state.reshape((-1,1))
               x2 = prim['x0'][0,0]
               x3 = x1 - prim['x0'][0,0]
               x4 = np.matmul(np.linalg.inv(np.diag([4, 0.02, 4, 4])), (x1-prim['x0'][0,0]))
               self.extended_state = (np.vstack((x1,x2,x3,x4)))[:,0] # initial state, consisting of actual state and virtual states for the controller
           k = int(prim_progress * N) # calculate primitive waypoint

           dist = get_disturbance()
           q1 = prim['K'][0,0][k,0].reshape((-1, 1), order='F')
           q2 = 0.5 * (prim['x_ref'][0,0][:,k+1] + prim['x_ref'][0,0][:,k]).reshape(-1,1)
           q3 = prim['u_ref'][0,0][:,k].reshape(-1,1)
           q4 = prim['u_ref'][0,0][:,k].reshape(-1,1)
           q5 = np.matmul(G_u, prim['alpha'][0,0][k*nu:(k+1)*nu]).reshape((-1,1), order='F')
           q = np.vstack((q1,q2,q3,q4,q5)) # parameters for the controller
           self.extended_state = odeint(func = prim_state_dot, y0 = self.extended_state, t= [0, dt], args=(dist, q))[-1, :]
           self.state = self.extended_state[0:4]
           # update alive time
           self.alive_time += dt
           # update progress
           prim_progress = prim_progress + dt / t_end
           self.prim_queue.replace_top((prim_id, prim_progress))

# TESTING
#prim_id = 0
#prim = mat['MA3'][prim_id,0] # the primitive corresponding to the primitive number
#x0 = np.array(prim['x0'][0,0]) + np.matmul(np.diag([4, 0.02, 4, 4]),2*np.random.rand(4,1)-np.ones((4,1))) # random state in initial set TODO: incorporate self.state
#print(x0)
#my_car = KinematicCar(init_state = np.reshape(x0, (-1, 1)))
#progress = 0
#while my_car.prim_progress <= 1:
#    my_car.prim_next(prim_id = 0, dt = 0.1)
