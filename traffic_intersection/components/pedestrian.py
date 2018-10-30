#!/usr/local/bin/python
# Pedestrian Class
# Tung M. Phan
# California Institute of Technology
# June 5, 2018

import imageio
import os
import numpy as np
from PIL import Image
import scipy.integrate as integrate
dir_path = os.path.dirname(os.path.realpath(__file__))
from prepare.queue import Queue

medic = dir_path + '/imglib/pedestrians/medic' + '.png'
all_pedestrian_types = {'1','2','3','4','5','6'}

class Pedestrian:
    def __init__(self,
                 init_state = [0,0,0,0], # (x, y, theta, gait)
                 number_of_gaits = 6,
                 gait_length = 4,
                 gait_progress = 0,
                 film_dim = (1, 6),
                 prim_queue = None, # primitive queue
                 pedestrian_type = '3',
                 name = None,
                 age = 20):
        """
        Pedestrian class
        """
        # init_state: initial state by default (x = 0, y = 0, theta = 0, gait = 0)
        self.alive_time = 0
        self.is_dead = False
        self.state = np.array(init_state, dtype="float")
        self.number_of_gaits = film_dim[0] * film_dim[1]
        self.gait_length = gait_length
        self.gait_progress = gait_progress
        self.film_dim = film_dim
        self.name = name
        self.age = age
        if prim_queue == None:
            self.prim_queue = Queue()
        else:
            prim_queue = prim_queue
        self.fig = dir_path + '/imglib/pedestrians/walking' + pedestrian_type + '.png'

    def next(self, inputs, dt):
        """
        The pedestrian advances forward
        """
        if self.is_dead:
            if self.fig != medic:
                self.fig = medic
        else:
            dee_theta, vee = inputs
            self.state[2] += dee_theta # update heading of pedestrian
            self.state[0] += vee * np.cos(self.state[2]) * dt # update x coordinate of pedestrian
            self.state[1] += vee * np.sin(self.state[2]) * dt # update y coordinate of pedestrian
            distance_travelled = vee * dt # compute distance travelled during dt
            gait_change = (self.gait_progress + distance_travelled / self.gait_length) // 1 # compute number of gait change
            self.gait_progress = (self.gait_progress + distance_travelled / self.gait_length) % 1
            self.state[3] = int((self.state[3] + gait_change) % self.number_of_gaits)
            self.alive_time += dt

    def visualize(self):
        # convert gait number to i, j coordinates of subfigure
        current_gait = self.state[3]
        i = current_gait % self.film_dim[1]
        j = current_gait // self.film_dim[1]
        img = Image.open(self.fig)
        width, height = img.size
        sub_width = width/self.film_dim[1]
        sub_height = height/self.film_dim[0]
        lower = (i*sub_width, (j-1)*sub_height)
        upper = ((i+1)*sub_width, j*sub_height)
        area = (lower[0], lower[1], upper[0], upper[1])
        cropped_img = img.crop(area)
        return cropped_img

    def extract_primitive(self):
       #TODO: rewrite the comment below
       """
       This function updates the primitive queue and picks the next primitive to be applied. When there is no more primitive in the queue, it will
       return False

       """
       while self.prim_queue.len() > 0:
           if self.prim_queue.top()[1] < 1: # if the top primitive hasn't been exhausted
               prim_data, prim_progress = self.prim_queue.top() # extract it
               return prim_data, prim_progress
           else:
               self.prim_queue.pop() # pop it
       return False

    def prim_next(self, dt):
        if self.extract_primitive() == False: # if there is no primitive to use
            self.next((0, 0), dt)
        else:
            prim_data, prim_progress = self.extract_primitive() # extract primitive data and primitive progress from prim
            start, finish, vee = prim_data # extract data from primitive
            x = finish[0] - start[0]
            y = finish[1] - start[1]
            total_distance = np.linalg.norm(np.array([x, y]))
            if prim_progress == 0: # ensure that starting position is correct at start of primitive
                self.state[0] = start[0]
                self.state[1] = start[1]
            if start == finish: #waiting mode
                remaining_distance = 0
                self.state[3] = 0 # reset gait
                if self.prim_queue.len() > 1: # if current not at last primitive
                    last_prim_data, last_prim_progress = self.prim_queue.bottom() # extract last primitive
                    last_start, last_finish, vee = last_prim_data
                    dx_last = last_finish[0] - self.state[0]
                    dy_last = last_finish[1] - self.state[1]
                    heading = np.arctan2(dy_last,dx_last)
                    if self.state[2] != heading:
                        self.state[2] = heading
            else: # if in walking mode
                dx = finish[0] - self.state[0]
                dy = finish[1] - self.state[1]
                remaining_distance = np.linalg.norm(np.array([dx, dy]))
                heading = np.arctan2(dy,dx)
                if self.state[2] != heading:
                    self.state[2] = heading
            if vee * dt > remaining_distance and remaining_distance != 0:
                self.next((0, remaining_distance/dt), dt)
            else:
                self.next((0, vee), dt)
            if total_distance != 0:
                prim_progress += dt / (total_distance / vee)
            self.prim_queue.replace_top((prim_data, prim_progress)) # update primitive queue

    # if the pedestrian isn't going fast enough, increase speed to cross street before light turns red
    def walk_faster(self, remaining_time):
        person_xy = (self.state[0], self.state[1])
        prim_data, prim_progress = self.prim_queue.top()
        start, finish, vee = prim_data
        dx = finish[0] - self.state[0]
        dy = finish[1] - self.state[1]
        remaining_distance = np.linalg.norm(np.array([dx, dy]))
        vee_max = 50
        if (remaining_distance / vee) > remaining_time:
            vee = remaining_distance / remaining_time
            if (vee <= vee_max):
                self.prim_queue.replace_top(((start, finish, vee), prim_progress))

    # cross the street if light is green, or if the pedestrian just crossed the street, continue walking and ignore traffic light color
    def continue_walking(self, walk_sign, lane1, lane2, direction, remaining_time):
        person_xy = (self.state[0], self.state[1])
        theta = self.state[2]
        if person_xy in (lane1 + lane2) and walk_sign:
            prim_data, prim_progress = self.prim_queue.get_element_at_index(-2)
            start, finish, vee = prim_data
            dx = finish[0] - start[0]
            dy = finish[1] - start[1]
            remaining_distance = np.linalg.norm(np.array([dx, dy]))
            vee_max = 50
            if (remaining_distance / vee) > remaining_time:
                vee = remaining_distance / remaining_time
            return (vee <= vee_max)
        elif (person_xy == lane1[0] or person_xy == lane2[0]) and theta == direction[0]:
            return True
        elif (person_xy == lane1[1] or person_xy == lane2[1]) and theta == direction[1]:
            return True
        else:
            return False
