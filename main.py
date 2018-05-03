# Simulation Plaform for Street Intersection Controller
# Tung M. Phan
# California Institute of Technology
# May 2, 2018

import os
import random
import car
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

dir_path = os.path.dirname(os.path.realpath(__file__))
# create figure
fig = plt.figure()
# remove axis from figure
plt.axis("off")
intersection_fig = dir_path + "/imglib/intersection.png"
blue_car_fig = dir_path + "/imglib/blue_car.png"
gray_car_fig = dir_path + "/imglib/gray_car.png"
car_scale_factor = 0.12

def add_car(states, car_type='blue'):
    x, y, theta = states[0], states[1], states[2]
    if car_type == 'blue':
        car = Image.open(blue_car_fig)
    elif car_type == 'gray':
        car = Image.open(gray_car_fig)
    # set expand=True so as to disable cropping of output image
    car = car.rotate(theta, expand=True)
    scaled_car_size  =  tuple([int(car_scale_factor * i) for i in car.size])
    # rescale car 
    car = car.resize(scaled_car_size, Image.ANTIALIAS)
    background.paste(car, (x,y), car)

#add_car((500,220,50))
#add_car((545,620,90), 'gray')
#background.show()
frames = []
def animate(i):
    plt.plot(1,1)


for i in range(0,100):
    # create new background
    background = Image.open(intersection_fig)
    add_car((225+i*2,360,0))
    frames.append([plt.imshow(background, animated = True)])
ani = animation.ArtistAnimation(fig, frames, interval=10, blit=True, repeat_delay=1)
#ani = animation.ArtistAnimation(fig, animate, interval=10, blit=True, repeat_delay=1)
plt.show()
