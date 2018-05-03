# Kinematic Car Class
# Tung M. Phan
# California Institute of Technology
# May 3, 2018
from numpy import cos, sin, tan
import scipy.integrate as integrate
import numpy as np

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
                 nu_max = 0.5, # maximum steering input in radians
                 nu_min = -0.5): # minimum steering input in radians)
                     self.init_state = np.array(init_state, dtype="float")
                     self.params = (L, a_max, a_min, nu_max, nu_min)
                     self.alive_time = 0
                     self.state = self.init_state
    def state_dot(self,
                  state,
                  t,
                  a,
                  nu):
       """
       state_dot is a Function that defines the system dynamics

       """
       (L, a_max, a_min, nu_max, nu_min) = self.params
       dstate_dt = np.zeros(np.shape(state))
       dstate_dt[0] = saturation_filter(a, a_max, a_min)
       dstate_dt[1] = state[0] / L * tan(saturation_filter(nu, nu_max, nu_min))
       dstate_dt[2] = state[0] * cos(state[1])
       dstate_dt[3] = state[0] * sin(state[1])
       return dstate_dt

    def next(self, inputs, dt):
       # take only the real part of the solution
       self.state = integrate.odeint(self.state_dot, self.state, t=(0, dt), args=inputs)[1]
       self.alive_time += dt


car = KinematicCar()
dt = 0.1
t = 0
while t < 1:
    car.next(inputs=(1,1),dt = dt)
    print(car.state)
    t += dt

