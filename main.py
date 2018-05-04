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
intersection_fig = dir_path + "/imglib/intersection.png"
blue_car_fig = dir_path + "/imglib/blue_car.png"
gray_car_fig = dir_path + "/imglib/gray_car.png"
car_scale_factor = 0.12

def draw_car(vehicle):
    vee, theta, x, y = vehicle.state
    # convert angle to degrees and positive counter-clockwise
    theta = -theta/np.pi * 180
    if vehicle.color  == 'blue':
        vehicle_fig = Image.open(blue_car_fig)
    elif vehicle.color  == 'gray':
        vehicle_fig = Image.open(gray_car_fig)
    # set expand=True so as to disable cropping of output image
    vehicle_fig = vehicle_fig.rotate(theta, expand=True)
    scaled_vehicle_fig_size  =  tuple([int(car_scale_factor * i) for i in vehicle_fig.size])
    # rescale car 
    vehicle_fig = vehicle_fig.resize(scaled_vehicle_fig_size, Image.ANTIALIAS)
    background.paste(vehicle_fig, (int(x),int(y)), vehicle_fig)

# creates figure
fig = plt.figure()
# turn on/off axes
plt.axis('on')


# Artist Animation option is used to generate offline movies - implemented here as a backup
use_artist_animation = False
if use_artist_animation:
    for i in range(0,100):
        # create new background
        background = Image.open(intersection_fig)
        # test values - should be changed according to needs
        draw_car((225+i*2,360,0))
        frames.append([plt.imshow(background, animated = True)])
    ani = animation.ArtistAnimation(fig, frames, interval=10, blit=True, repeat_delay=1)
    plt.show()
else:
    # sampling time
    dt = 0.1
    # creates cars
    car_1 = car.KinematicCar(init_state=(0,np.pi/2,200,300), L=70)
    car_2 = car.KinematicCar(init_state=(1,np.pi/4,300,200), color='gray', L=200)
    cars = [car_1, car_2]
    background = Image.open(intersection_fig)

    def init():
        stage = plt.imshow(background, origin="lower") # this origin option flips the y-axis
        return stage, # notice the comma is required to make returned object iterable (a requirement of FuncAnimation)

    def animate(i):
        """ online frame update """
        global dt, background
        # update background
        background = Image.open(intersection_fig)
        x_lim, y_lim = background.size
        # update cars
        for vehicle in cars:
            nu = np.sin(i*0.01)
            vehicle.next((15, nu),dt)
            if (vehicle.state[2] <= x_lim and vehicle.state[3] <= y_lim):
                draw_car(vehicle)
        stage = plt.imshow(background, origin="lower") # this origin option flips the y-axis
        return stage, # notice the comma is required to make returned object iterable (a requirement of FuncAnimation)

    from time import time
    t0 = time()
    animate(0)
    t1 = time()
    interval = (t1 - t0)
    ani = animation.FuncAnimation(fig, animate, frames=300, interval=interval, blit=True, init_func = init)
    plt.show()
