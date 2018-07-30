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


def generate_walking_gif():
    pedestrian_fig = dir_path + "/imglib/pedestrians/walking3.png"
    film_dim = (1, 6)  # dimension of the film used for animation
    img = Image.open(pedestrian_fig)
    width, height = img.size
    sub_width = width/film_dim[1]
    sub_height = height/film_dim[0]
    images = []
    for j in range(0, film_dim[0]):
        for i in range(0, film_dim[1]):
            lower = (i*sub_width,  j*sub_height)
            upper = ((i+1)*sub_width, (j+1)*sub_height)
            area = (lower[0], lower[1], upper[0], upper[1])
            cropped_img = img.crop(area)
            cropped_img = np.asarray(cropped_img)
            images.append(cropped_img)
    imageio.mimsave('movie.gif', images, duration=0.1)


class Pedestrian:
    def __init__(self,
                 init_state=[0, 0, 0, 0],  # (x, y, theta, gait)
                 number_of_gaits=6,
                 gait_length=4,
                 gait_progress=0,
                 film_dim=(1, 6),
                 prim_queue=None,  # primitive queue
                 pedestrian_type='3'):  # three types 1 or 2 or 3
        """
        Pedestrian class
        """
        # init_state: initial state by default (x = 0, y = 0, theta = 0, gait = 0)
        self.alive_time = 0
        self.state = np.array(init_state, dtype="float")
        self.number_of_gaits = film_dim[0] * film_dim[1]
        self.gait_length = gait_length
        self.gait_progress = gait_progress
        self.film_dim = film_dim
        if prim_queue == None:
            self.prim_queue = Queue()
        else:
            prim_queue = prim_queue
        self.fig = dir_path + '/imglib/pedestrians/walking' + pedestrian_type + '.png'

    def next(self, inputs, dt):
        """
        The pedestrian advances forward
        """
        dee_theta, vee = inputs
        self.state[2] += dee_theta  # update heading of pedestrian
        # update x coordinate of pedestrian
        self.state[0] += vee * np.cos(self.state[2]) * dt
        # update y coordinate of pedestrian
        self.state[1] += vee * np.sin(self.state[2]) * dt
        distance_travelled = vee * dt  # compute distance travelled during dt
        gait_change = (self.gait_progress + distance_travelled /
                       self.gait_length) // 1  # compute number of gait change
        self.gait_progress = (self.gait_progress +
                              distance_travelled / self.gait_length) % 1
        self.state[3] = int((self.state[3] + gait_change) %
                            self.number_of_gaits)

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
        # TODO: rewrite the comment below
        """
        This function updates the primitive queue and picks the next primitive to be applied. When there is no more primitive in the queue, it will
        return False

        """
        while self.prim_queue.len() > 0:
            # if the top primitive hasn't been exhausted
            if self.prim_queue.top()[1] < 1:
                prim_data, prim_progress = self.prim_queue.top()  # extract it
                return prim_data, prim_progress
            else:
                self.prim_queue.pop()  # pop it
        return False

    def prim_next(self, dt):
        if self.extract_primitive() == False:  # if there is no primitive to use
            self.next((0, 0), dt)
        else:
            # extract primitive data and primitive progress from prim
            prim_data, prim_progress = self.extract_primitive()
            start, finish, t_end = prim_data  # extract data from primitive
            if prim_progress == 0:  # ensure that starting position is correct at start of primitive
                self.state[0] = start[0]
                self.state[1] = start[1]
            if start == finish:  # waiting mode
                remaining_distance = 0
                self.state[3] = 0  # reset gait
                if self.prim_queue.len() > 1:  # if current not at last primitive
                    last_prim_data, last_prim_progress = self.prim_queue.bottom()  # extract last primitive
                    last_start, last_finish, last_t_end = last_prim_data
                    dx_last = last_finish[0] - self.state[0]
                    dy_last = last_finish[1] - self.state[1]
                    heading = np.arctan2(dy_last, dx_last)
                    if self.state[2] != heading:
                        self.state[2] = heading
            else:  # if in walking mode
                dx = finish[0] - self.state[0]
                dy = finish[1] - self.state[1]
                heading = np.arctan2(dy, dx)
                if self.state[2] != heading:
                    self.state[2] = heading
                remaining_distance = np.linalg.norm(np.array([dx, dy]))
            remaining_time = (1-prim_progress) * t_end
            vee = remaining_distance / remaining_time
            self.next((0, vee), dt)
            prim_progress += dt / t_end
            self.prim_queue.replace_top(
                (prim_data, prim_progress))  # update primitive queue

#my_pedestrian = Pedestrian()
#dt = 0.1
##from matplotlib import pyplot as plt
##box = (70, 70, 30, 30)
# my_pedestrian.visualize()
# while True:
#    my_pedestrian.next((0.1, 0.05), dt)
#    print(my_pedestrian.state)
