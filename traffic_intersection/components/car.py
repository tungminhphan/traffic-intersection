# Kinematic Car Class
# Tung M. Phan
# California Institute of Technology
# May 3, 2018

import os
import sys
sys.path.append("..")
import scipy.io
import numpy as np
from primitives.prim_car import prim_state_dot
from components.aux.tire_data import get_tire_data
from scipy.integrate import odeint
from numpy import cos, sin, tan, arctan2, sqrt
main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
primitive_data = main_dir + '/primitives/MA3.mat'
from prepare.queue import Queue
from PIL import Image
from assumes.disturbance import get_disturbance
import assumes.params as params
from math import pi

dir_path = os.path.dirname(os.path.realpath(__file__))
blue_car_fig = dir_path + "/imglib/cars/blue_car.png"
gray_car_fig = dir_path + "/imglib/cars/gray_car.png"
car_figs = {
    "blue": blue_car_fig,
    "gray": gray_car_fig
}
mat = scipy.io.loadmat(primitive_data)


def saturation_filter(u, u_max, u_min):
    ''' saturation_filter Helper Function

        the output is equal to u_max if u >= u_max, u_min if u <= u_min, and u otherwise
    '''
    return max(min(u, u_max), u_min)


#def force_saturation_function(sigma, tire_data):
#    ''' Force saturation function used in the calculation of tire traction forces
#    Inputs:
#    sigma - the composite slip
#    tire_data - tire parameters
#
#    Output:
#    f - composite force coefficient such tat f = F_c / (mu * F_z)
#    '''
#    C_1 = tire_data['C1']
#    C_2 = tire_data['C2']
#    C_3 = tire_data['C3']
#    C_4 = tire_data['C4']
#    f =  (C_1 * sigma**3 + C_2 * sigma**2 + (4/np.pi) * sigma)/(C_1 * sigma**3 + C_3 * sigma ** 2 + C_4 * sigma + 1)
#    return f

class KinematicCar:
    '''Kinematic car class

    init_state is [vee, theta, x, y], where vee, theta, x, y are the velocity, orientation, and
    coordinates of the car respectively
    '''
    def __init__(self, init_state=[0, 0, 0, 0],
                 L=50,  # length of vehicle in pixels
                 a_max=9.81,  # maximum acceleration of vehicle
                 a_min=-9.81,  # maximum deceleration of vehicle
                 nu_max=0.5,  # maximum steering input in radians/sec
                 nu_min=-0.5,  # minimum steering input in radians/sec)
                 vee_max=100,  # maximum velocity
                 is_honking=False,  # the car is honking
                 color='blue',  # color of the car
                 plate_number=None,  # license plate number
                 # queue of primitives, each item in the queue has the form (prim_id, prim_progress) where prim_id is the primitive ID and prim_progress is the progress of the primitive)
                 prim_queue=None,
                 fuel_level=float('inf')):  # TODO: fuel level of the car - FUTURE FEATURE)
        if color != 'blue' and color != 'gray':
            raise Exception("Color must either be blue or gray!")
        self.params = (L, a_max, a_min, nu_max, nu_min, vee_max)
        self.alive_time = 0
        self.plate_number = plate_number
        self.state = np.array(init_state, dtype='float')
        self.color = color
        # extended state required for Bastian's primitive computation
        self.extended_state = None
        self.is_honking = is_honking
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
            dstate_dt[1] = state[0] / L * \
                tan(saturation_filter(nu, nu_max, nu_min))
        dstate_dt[2] = state[0] * cos(state[1])
        dstate_dt[3] = state[0] * sin(state[1])
        return dstate_dt

    def toggle_honk(self):
        self.is_honking = not self.is_honking

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
        self.state = odeint(self.state_dot, self.state,
                            t=(0, dt), args=(a, nu))[1]
        # fuel decreases linearly with acceleration/deceleration
        self.fuel_level -= np.abs(a) * dt
        # update alive time
        self.alive_time += dt

        # TODO: temporary fix to floating problem
        if a == 0:
            self.state[0] = np.sign(self.state[0]) * abs(self.state[0]) * dt

    def extract_primitive(self):
        # TODO: rewrite the comment below
        """
        This function updates the primitive queue and picks the next primitive to be applied. When there is no more primitive in the queue, it will
        return False

        """
        while self.prim_queue.len() > 0:
            # if the top primitive hasn't been exhausted
            if self.prim_queue.top()[1] < 1:
                prim_id, prim_progress = self.prim_queue.top()  # extract it
                return prim_id, prim_progress
            else:
                self.prim_queue.pop()  # pop it
        return False

    def prim_next(self, dt):
        """
        updates with primitive, if no primitive available, update with next with zero inputs
        Inputs:
        dt: integration time

        Outputs:
        None - the states of the car will get updated
        """
        # TODO: implement compatibility check with primitive to make sure that the params & dynamics match (essentially comparing prim_car and the
        # kinematic model)
        if self.extract_primitive() == False:  # if there is no primitive to use
            self.next((0, 0), dt)
        else:
            prim_id, prim_progress = self.extract_primitive()
            # load primitive data TODO: make this portion of the code more automated
            if prim_id > -1:
                # the primitive corresponding to the primitive number
                prim = mat['MA3'][prim_id, 0]
                # extract duration of primitive
                t_end = prim['t_end'][0, 0][0, 0]
                # number of subintervals encoded in primitive
                N = prim['K'][0, 0].shape[0]
                # this diagonal matrix encodes the size of input set (a constraint)
                G_u = np.diag([175, 1.29])
                nu = 2  # number of inputs
                nx = 4  # number of states

                if prim_progress == 0:  # compute initial extended state
                    x1 = self.state.reshape((-1, 1))
                    x2 = prim['x0'][0, 0]
                    x3 = x1 - prim['x0'][0, 0]
                    x4 = np.matmul(np.linalg.inv(
                        np.diag([4, 0.02, 4, 4])), (x1-prim['x0'][0, 0]))
                    # initial state, consisting of actual state and virtual states for the controller
                    self.extended_state = (np.vstack((x1, x2, x3, x4)))[:, 0]
                k = int(prim_progress * N)  # calculate primitive waypoint

                dist = get_disturbance()
                q1 = prim['K'][0, 0][k, 0].reshape((-1, 1), order='F')
                q2 = 0.5 * (prim['x_ref'][0, 0][:, k+1] +
                            prim['x_ref'][0, 0][:, k]).reshape(-1, 1)
                q3 = prim['u_ref'][0, 0][:, k].reshape(-1, 1)
                q4 = prim['u_ref'][0, 0][:, k].reshape(-1, 1)
                q5 = np.matmul(
                    G_u, prim['alpha'][0, 0][k*nu:(k+1)*nu]).reshape((-1, 1), order='F')
                # parameters for the controller
                q = np.vstack((q1, q2, q3, q4, q5))
                self.extended_state = odeint(func=prim_state_dot, y0=self.extended_state, t=[
                                             0, dt], args=(dist, q))[-1, :]
                self.state = self.extended_state[0:4]
                # update alive time
                self.alive_time += dt
                # update progress
                prim_progress = prim_progress + dt / t_end
                self.prim_queue.replace_top((prim_id, prim_progress))
            else:  # if is stopping primitive
                self.next((0, 0), dt)

class DynamicCar(KinematicCar): # bicycle 5 DOF model
    def __init__(self, 
                m = 2000, # mass of the vehile in kilograms
                m_w = 10, # mass of one tire
                L_r = 1.2, # distance in meters from rear axle to center of mass
                L_f = 1.2, # distance in meters from front axle to center of mass
                h = 0.4, # height of center of mass in meters
                tire_designation = '155SRS13', # tire specifications
                init_dyn_state = np.zeros(7), # initial dynamical state of vehicle
                car_width = 1.2,  # car width
                R_w = 0.25): # the radius of the vehicle's wheel
        KinematicCar.__init__(self)
        self.m = m
        self.L = self.params[0]
        self.L_r = L_r
        self.L_f = L_f
        self.h = h
        self.R_w = R_w
        self.m_w = m_w
        self.tire_data = get_tire_data(tire_designation)
        self.dyn_state =  init_dyn_state
        self.car_width = car_width
        self.I_w = 0.5 * m_w * R_w ** 2 # approximated as the moment of inertia around z-axis of a thin disk / cylinder of any length with radius R_w and mass m
        self.I_z = 1/12. * m * (self.L**2 + self.car_width**2)

    def state_dot(self, t, delta_f, delta_r, T_af, T_ar, T_bf, T_br):
        # state = [v_x, v_y, psi, w_f, w_r, X, Y]
        state = self.dyn_state
        v_x = state[0]
        v_y = state[1]
        psi = state[2]
        w_f = state[3]
        w_r = state[4]
        X = state[5]
        Y = state[6]

        m = self.m
        L_r = self.L_r
        L_f = self.L_f
        h = self.h
        R_w = self.R_w

        I_w = self.I_w
        I_z = self.I_z
        g = params.g

        # iteratively solving for a_x, F_xf, F_xr: the instantaneous longitudinal acceleration, and longitudinal traction forces for the front and rear tires respectively
        a_x = 0 # first guess for longitudinal acceleration
        F_xf_guess = 0 # first guess for front tire longitudinal traction
        F_xr_guess = 0 # first guess for rear tire longitudinal traction
        F_yf_guess = 0 # first guess for front tire tangential traction
        F_yr_guess = 0 # first guess for rear tire tangential traction
        r_guess= 0 # first guess for yaw rate

        F_xf = F_xf_guess 
        F_xr = F_xr_guess 
        F_yf = F_yf_guess
        F_yr = F_yr_guess
        r = r_guess

        iter_error = float('inf')


        while iter_error > 0.01: # 
            iter_errors = []
            # equation (1)
            rhs = -F_xf * cos(delta_f) - F_yf * sin(delta_f) - F_xr * cos(delta_r) - F_yr * sin(delta_r)
            vdot_x = rhs / m + v_y * r # state 1
            # equation (2)
            rhs = F_yf * cos(delta_f) - F_xf * sin(delta_f) + F_yr * cos(delta_r) - F_xr * sin(delta_r)
            vdot_y = rhs / m  - v_x * r # state 2
            # equation (3)
            rhs = L_f * (F_yf * cos(delta_f) - F_xf * sin(delta_f)) - L_r * (F_yr * cos(delta_r) - F_xr * sin(delta_r))
            r = rhs / I_z
            # equation (4)
            rhs = F_xf * R_w - T_bf + T_af
            wdot_f =  rhs / I_w # state 3
            # equation (5)
            rhs = F_xr * R_w - T_br + T_ar
            wdot_r =  rhs / I_w # state 4
            # equation (6)
            psi_dot = r # state 5
            # equation (7)
            v_X = v_x * cos(psi) - v_y * sin(psi) # state 6
            # equation (8)
            v_Y = v_x * sin(psi) + v_y * cos(psi) # state 7
            # equation (11)
            alpha_f = arctan2(v_y + L_f * r, v_x) - delta_f
            # equation (12)
            alpha_r = arctan2(v_y - L_r * r, v_x) - delta_r
            # equation (13)
            V_tf = sqrt((v_y + L_f * r)**2 + v_x**2)
            # equation (14)
            V_tr = sqrt((v_y - L_r * r)**2 + v_x**2)
            # equation (15)
            v_wxf = V_tf * cos(alpha_f)
            # equation (16)
            v_wxr = V_tr * cos(alpha_r)
            # equation (17)
            S_af = ((v_wxf - w_f * R_w) / v_wxf)
            # equation (18)
            S_ar = ((v_wxr - w_r * R_w) / v_wxr)
            # equation (9)
            F_zf = (m * g * L_r - m * a_x * h) / (L_f + L_r)
            # equation (10)
            F_zr = (m * g * L_f - m * a_x * h) / (L_f + L_r)

            # compute traction force estimates
            F_xf, F_yf = self.get_traction(F_xf, F_zf, S_af, alpha_f)
            F_xr, F_yr = self.get_traction(F_xr, F_zr, S_ar, alpha_r)

            # recompute errors
            error_a_x = abs(vdot_x - a_x)
            iter_errors.append(error_a_x)
            error_F_xf = abs(F_xf_guess - F_xf)
            iter_errors.append(error_F_xf)
            error_F_xr = abs(F_xr_guess - F_xr)
            iter_errors.append(error_F_xr)
            error_F_yf = abs(F_yf_guess - F_yf)
            iter_errors.append(error_F_yf)
            error_F_yr = abs(F_yr_guess - F_yr)
            iter_errors.append(error_F_yr)
            error_r = abs(r_guess - r)
            iter_errors.append(error_r)

            iter_error = max(iter_errors)
            a_x = vdot_x
            F_xf_guess = F_xf
            F_xr_guess = F_xr
            F_yf_guess = F_yf
            F_yr_guess = F_yr
            r_guess = r

        dstate_dt = [vdot_x, vdot_y, psi_dot, wdot_f, wdot_r, v_X, v_Y]
        return dstate_dt

    def next(self):
        pass

    def get_traction(self, F_x, F_z, S, alpha): # longitudinal slip, slip angle, F_x, normal load
        tire_data = self.tire_data
        T_w = tire_data['T_w']
        T_p = tire_data['T_p']
        F_ZT = tire_data['F_ZT']
        C_1 = tire_data['C_1']
        C_2 = tire_data['C_2']
        C_3 = tire_data['C_3']
        C_4 = tire_data['C_4']
        A_0 = tire_data['A_0']
        A_1 = tire_data['A_1']
        A_2 = tire_data['A_2']
        A_3 = tire_data['A_3']
        A_4 = tire_data['A_4']
        K_a = tire_data['K_a']
        K_1 = tire_data['K_1']
        CS_FZ = tire_data['CS_FZ']
        mu_o = tire_data['mu_o']

        K_mu = 0.124
        Y_camber = 0 # camber angle

        # equation 11 & 12, tire contact patch length
        a_po = (0.0768 * sqrt(F_z * F_ZT)) / (T_w * (T_p + 5))
        a_p = a_po * (1 - K_a * F_x / F_z)

        # equation 13, 14 lateral and longitudianl stiffness coeffs 
        K_s = (2 / a_po**2) * (A_0 + (A_1 * F_z) - ((A_1 / A_2) * F_z**2)) 
        K_c = (2 / a_po**2) * F_z * CS_FZ

        # equation 15, composite slip calculation
        sigma = (pi * a_p**2) / ( 8 * mu_o * F_z) * sqrt((K_s**2 * tan(alpha)**2) + (K_c**2 * (S / (1 - S))**2))

        # equation 10 composite force, saturation function
        f_of_sigma = ((C_1 * sigma**3) + (C_2 * sigma**2) + (4 / pi * sigma)) / ((C_1 * sigma**3) + (C_3 * sigma**2) + (C_4 * sigma) + 1)

        # equation 18 & 19 modified Long. Stiff Coeff and Tire/Rd coeff friction
        K_c_prime = K_c + (K_s - K_c) * sqrt(sin(alpha)**2 + S**2 * cos(alpha)**2)
        mu = mu_o * (1 - K_mu * sqrt(sin(alpha)**2 + S**2 * cos(alpha)**2) )

        # equation 16 & 17 Normalized Lateral and Long Force
        F_y = (f_of_sigma * K_s * tan(alpha) / sqrt(K_s**2 * tan(alpha)**2 + K_c_prime**2 * S**2) + Y_camber)
        F_y /= mu * F_z
        F_x = f_of_sigma * K_c_prime * S / sqrt(K_s**2 * tan(alpha)**2 + K_c_prime**2 * S**2)
        F_x /= mu * F_z

        return F_x, F_y

# TESTING
# state = [v_x, v_y, psi, w_f, w_r, X, Y]
dyn_car = DynamicCar(init_dyn_state = np.array([0.1,0.1,0,0.1,0.1,100,100]))
#state_dot(self, t, delta_f, delta_r, T_af, T_ar, T_bf, T_br):
print('state_dot: a_x, a_y, r, d_w_f, d_w_r, v_X, v_Y')
print(dyn_car.state_dot(t = 0, delta_f = 0, delta_r = 0, T_af = 50, T_ar = 0, T_bf = 0, T_br =0))
