# Kinematic Car Class
# Tung M. Phan
# California Institute of Technology
# May 3, 2018
from numpy import cos, sin, tan
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

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
                 L = 3, # length of vehicle in meters
                 a_max = 9.81, # maximum acceleration of vehicle
                 a_min = -9.81, # maximum deceleration of vehicle
                 nu_max = 0.5, # maximum steering input in radians/sec
                 nu_min = -0.5, # minimum steering input in radians/sec)
                 color = 'blue', # color of the car)
                 fuel_level = float('inf')): # fuel level of the car)
                     if color != 'blue' and color != 'gray':
                         raise Exception("Color must either be blue or gray!")
                     self.init_state = np.array(init_state, dtype="float")
                     self.params = (L, a_max, a_min, nu_max, nu_min)
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
       state_dot is a Function that defines the system dynamics

       Inputs
       state: current state
       t: current time
       a: acceleration input
       nu: steering input

       """
       (L, a_max, a_min, nu_max, nu_min) = self.params
       dstate_dt = np.zeros(np.shape(state))
       dstate_dt[0] = saturation_filter(a, a_max, a_min)
       dstate_dt[1] = state[0] / L * tan(saturation_filter(nu, nu_max, nu_min))
       dstate_dt[2] = state[0] * cos(state[1])
       dstate_dt[3] = state[0] * sin(state[1])
       return dstate_dt

    def next(self, inputs, dt):
       """
       next is a Function that updates

       """
       # take only the real part of the solution
       a, nu = inputs
       # fuel decreases linearly with acceleration
       if self.fuel_level >= a*dt:
           self.state = integrate.odeint(self.state_dot, self.state, t=(0, dt), args=(a, nu))[1]
           self.fuel_level -= a * dt
       self.alive_time += dt
