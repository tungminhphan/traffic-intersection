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
pedestrian_fig = dir_path + "/imglib/pedestrians/walking2.png"
film_dim = (3, 5)
film_dim = (1, 8)
film_dim = (1, 6)

img = Image.open(pedestrian_fig)
width, height = img.size
sub_width = width/film_dim[1]
sub_height = height/film_dim[0]
images = []
for j in range(film_dim[0], 0, -1):
    for i in range(0, film_dim[1]):
        lower = (i*sub_width, (j-1)*sub_height)
        upper = ((i+1)*sub_width, j*sub_height)
        area = (lower[0], lower[1], upper[0], upper[1])
        cropped_img = img.crop(area)
        cropped_img = np.asarray(cropped_img)
        images.append(cropped_img)

imageio.mimsave('movie.gif', images, duration=0.1)
print('done')


class Pedestrian:
    def __init__(self, 
                 init_state = [0,0,0,0], 
                 number_of_gaits = 6, 
                 gait_length = 2,
                 gait_progress = 0):
        """
        Pedestrian class
        """
        # init_state: initial state by default (x = 0, y = 0, theta = 0, gait = 0)
        self.init_state = init_state
        self.alive_time = 0
        self.state = self.init_state
        self.number_of_gaits = 6
        self.gait_length = 2
        self.gait_progress = 0
   
    def next(self, inputs, dt):
        """
        The pedestrian takes one step forward
        """
        theta, v = inputs 
        self.state[0] += v * np.cos(theta) * dt # update x coordinate of pedestrian
        self.state[1] += v * np.sin(theta) * dt # update y coordinate of pedestrian
        self.state[2] = theta # update heading of pedestrian
        distance_travelled = v * dt # compute distance travelled during dt
        gait_change = (self.gait_progress + distance_travelled / self.gait_length) // 1
        self.gait_progress = (self.gait_progress + distance_travelled / self.gait_length) % 1
        self.state[3] = int((self.state[3] + gait_change) % self.number_of_gaits)

my_pedestrian = Pedestrian()
dt = 0.1
while True:
    my_pedestrian.next((0.1, 1), dt)
    print(my_pedestrian.state)

