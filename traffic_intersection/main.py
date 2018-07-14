#!/usr/local/bin/python
# Simulation Plaform for Street Intersection Controller
# Tung M. Phan
# California Institute of Technology
# May 2, 2018

import os
import sys
sys.path.append("..")
import traffic_intersection.components.car as car
import traffic_intersection.components.pedestrian as pedestrian
import traffic_intersection.components.traffic_signals as traffic_signals
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from time import time
from numpy import cos, sin, tan
import numpy as np
from PIL import Image
import random
import scipy.io

#TODO: clean up this section
dir_path = os.path.dirname(os.path.realpath(__file__))
primitive_data = dir_path + '/primitives/MA3.mat'
mat = scipy.io.loadmat(primitive_data)

intersection_fig = dir_path + "/components/imglib/intersection_states/intersection_"
car_scale_factor = 0.1 # scale for when L = 50
pedestrian_scale_factor = 0.32

def find_corner_coordinates(x_rel_i, y_rel_i, x_des, y_des, theta, square_fig):
    """
    This function takes an image and an angle then computes
    the coordinates of the corner (observe that vertical axis here is flipped)
    """
    w, h = square_fig.size
    theta = -theta
    if abs(w- h)>1:
        print("Warning: Figure has to be square!")
    x_corner_rel, y_corner_rel = -w/2, -h/2
    R = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    x_rel_f, y_rel_f = R.dot(np.array([[x_rel_i], [y_rel_i]]))
    # xy_unknown - xy_corner + xy_rel_f = xy_des
    return int(x_des - x_rel_f + x_corner_rel), int(y_des - y_rel_f + y_corner_rel)

def draw_car(vehicle):
    vee, theta, x, y = vehicle.state
    # convert angle to degrees and positive counter-clockwise
    theta_d = -theta/np.pi * 180
    vehicle_fig = vehicle.fig
    w_orig, h_orig = vehicle_fig.size
    # set expand=True so as to disable cropping of output image
    vehicle_fig = vehicle_fig.rotate(theta_d, expand = False)
    scaled_vehicle_fig_size  =  tuple([int(car_scale_factor * i) for i in vehicle_fig.size])
    # rescale car 
    #vehicle_fig = vehicle_fig.resize(scaled_vehicle_fig_size, Image.ANTIALIAS)
    vehicle_fig = vehicle_fig.resize(scaled_vehicle_fig_size) # disable antialiasing for better performance
    # at (full scale) the relative coordinates of the center of the rear axle w.r.t. the
    # center of the figure is -185
    print(vehicle_fig.size)
    x_corner, y_corner = find_corner_coordinates(-car_scale_factor * (w_orig/2-180), 0, x, y, theta, vehicle_fig)
    background.paste(vehicle_fig, (x_corner, y_corner), vehicle_fig)

def draw_pedestrian(pedestrian):
    x, y, theta, current_gait = pedestrian.state
    i = current_gait % pedestrian.film_dim[1]
    j = current_gait // pedestrian.film_dim[1]
    film_fig = Image.open(pedestrian.fig)
    scaled_film_fig_size  =  tuple([int(pedestrian_scale_factor * i) for i in film_fig.size])
    film_fig = film_fig.resize( scaled_film_fig_size)
    width, height = film_fig.size
    sub_width = width/pedestrian.film_dim[1]
    sub_height = height/pedestrian.film_dim[0]
    lower = (i*sub_width,j*sub_height)
    upper = ((i+1)*sub_width, (j+1)*sub_height)
    area = (int(lower[0]), int(lower[1]), int(upper[0]), int(upper[1]))
    person_fig = film_fig.crop(area)
    person_fig = person_fig.rotate(180-theta/np.pi * 180 + 90, expand = False)
    x_corner, y_corner = find_corner_coordinates(0., 0, x, y, theta,  person_fig)
    background.paste(person_fig, (int(x_corner), int(y_corner)), person_fig)

# creates figure
fig = plt.figure()
fig.add_axes([0,0,1,1]) # get rid of white border

# turn on/off axes
plt.axis('off')
# sampling time
dt = 0.1
# creates cars
prim_id = 0 # first primitive
prim = mat['MA3'][prim_id,0]
x0 = np.array(prim['x0'][0,0][:,0])
car_1a = car.KinematicCar(init_state = np.reshape(x0, (-1, 1))) # primitive car
car_1a.prim_queue.enqueue((prim_id, 0))
car_1a.prim_queue.enqueue((4, 0))
car_1a.prim_queue.enqueue((8, 0))
car_1a.prim_queue.enqueue((15, 0))
car_1a.prim_queue.enqueue((17, 0))

prim_id = 6 # first primitive
prim = mat['MA3'][prim_id,0]
x0 = np.array(prim['x0'][0,0][:,0])
car_1b = car.KinematicCar(init_state = np.reshape(x0, (-1, 1))) # primitive car
car_1b.prim_queue.enqueue((prim_id, 0))
car_1b.prim_queue.enqueue((8, 0))
car_1b.prim_queue.enqueue((13, 0))
car_1b.prim_queue.enqueue((14, 0))

prim_id = 25 # first primitive
prim = mat['MA3'][prim_id,0]
x0 = np.array(prim['x0'][0,0][:,0])
car_1c = car.KinematicCar(init_state = np.reshape(x0, (-1, 1))) # primitive car
car_1c.prim_queue.enqueue((prim_id, 0))
car_1c.prim_queue.enqueue((26, 0))
car_1c.prim_queue.enqueue((28, 0))

controlled_cars = [car_1a, car_1b, car_1c]
car_2 = car.KinematicCar(init_state=(60,np.pi/2,635,300), color='gray')
car_3 = car.KinematicCar(init_state=(50,0,0,250), color='gray')
car_4 = car.KinematicCar(init_state=(40,-np.pi,1000,520), color='gray')
enemy_cars = [car_2, car_3, car_4]
#
# delayed enemy_cars
car_6 = car.KinematicCar(init_state=(90,np.pi/2,635,0), color='gray')
car_7b = car.KinematicCar(init_state=(45,np.pi/2,565, 80), color='gray')
car_8 = car.KinematicCar(init_state=(80,-np.pi/2,430,762), color='gray')
car_9b = car.KinematicCar(init_state=(40,-np.pi/2,500,690), color='gray')
delay_time = 290
delayed_enemy_cars = [car_6, car_7b, car_8, car_9b]
# waiting enemy_cars
car_7 = car.KinematicCar(init_state=(0,np.pi/2,565, 80), color='gray')
car_9 = car.KinematicCar(init_state=(0,-np.pi/2,500,690), color='gray')
delay_time = 290
waiting_enemy_cars = [car_9, car_7]
# creates pedestrians
left_bottom = (0, 170)
right_bottom = (1062, 170)
left_top = (0, 590)
right_top = (1062, 590)

top_left = (355, 762)
top_right = (705, 762)
bottom_left = (355, 0)
bottom_right = (705, 0)

offset_wait = 25 # distance from the "pedestrian intersection" to waiting location
wait_top_left = (355, 590)
wait_top_left_vertical = (355, 590-offset_wait)
wait_top_left_horizontal = (355+offset_wait, 590)

wait_bottom_left = (355, 170)
wait_bottom_left_vertical = (355, 170+offset_wait)
wait_bottom_left_horizontal = (355+offset_wait, 170)

wait_top_right = (705, 590)
wait_top_right_vertical = (705, 590-offset_wait)
wait_top_right_horizontal = (705-offset_wait, 590)

wait_bottom_right = (705, 170)
wait_bottom_right_vertical = (705, 170+offset_wait)
wait_bottom_right_horizontal = (705-offset_wait, 170)

pedestrian_1 = pedestrian.Pedestrian(init_state=[705,590,-np.pi/2,0], pedestrian_type='1')
pedestrian_1.prim_queue.enqueue(((wait_top_right,wait_top_right, 15), 0))
pedestrian_1.prim_queue.enqueue(((wait_top_right,wait_top_left, 15), 0))
pedestrian_1.prim_queue.enqueue(((wait_top_left,wait_bottom_left, 15), 0))
pedestrian_1.prim_queue.enqueue(((wait_bottom_left,bottom_left, 10), 0))

pedestrian_2 = pedestrian.Pedestrian(init_state=[right_bottom[0],right_bottom[1], np.pi/2,0], pedestrian_type='2')
pedestrian_2.prim_queue.enqueue(((right_bottom,wait_bottom_right, 10), 0))
pedestrian_2.prim_queue.enqueue(((wait_bottom_right, bottom_right, 10), 0))

pedestrian_3 = pedestrian.Pedestrian(init_state=[left_bottom[0],left_bottom[1],np.pi/2,0], pedestrian_type='3')
pedestrian_3.prim_queue.enqueue(((left_bottom,wait_bottom_left, 10), 0))
pedestrian_3.prim_queue.enqueue(((wait_bottom_left,wait_bottom_left, 1), 0))
pedestrian_3.prim_queue.enqueue(((wait_bottom_left,wait_bottom_right, 10), 0))
pedestrian_3.prim_queue.enqueue(((wait_bottom_right,wait_bottom_right, 5), 0))
pedestrian_3.prim_queue.enqueue(((wait_bottom_right,wait_top_right, 12), 0))
pedestrian_3.prim_queue.enqueue(((wait_top_right,right_top, 10), 0))

pedestrian_4 = pedestrian.Pedestrian(init_state=[bottom_right[0],bottom_right[1],np.pi/2,0], pedestrian_type='4')
pedestrian_4.prim_queue.enqueue(((bottom_right,wait_bottom_right_vertical, 7), 0))
pedestrian_4.prim_queue.enqueue(((wait_bottom_right_vertical,wait_bottom_right_vertical, 20), 0))
pedestrian_4.prim_queue.enqueue(((wait_bottom_right_vertical,wait_top_right_vertical, 15), 0))
pedestrian_4.prim_queue.enqueue(((wait_top_right_vertical,top_right, 10), 0))

pedestrians = [pedestrian_1, pedestrian_2, pedestrian_3, pedestrian_4]
# create traffic lights
traffic_lights = traffic_signals.TrafficLights(3, 23, random_start = False)
horizontal_light = traffic_lights.get_states('horizontal', 'color')
vertical_light = traffic_lights.get_states('vertical', 'color')
background = Image.open(intersection_fig + horizontal_light + '_' + vertical_light + '.png')
def init():
    stage = plt.imshow(background, origin="lower",zorder=0) # this origin option flips the y-axis
    return stage, # notice the comma is required to make returned object iterable (a requirement of FuncAnimation)

def animate(i): # update animation by dt
    print(i)
    """ online frame update """
    global background
    # update traffic lights
    traffic_lights.update(dt)
    horizontal_light = traffic_lights.get_states('horizontal', 'color')
    vertical_light = traffic_lights.get_states('vertical', 'color')
    # update background
    # TODO: implement option to lay waypoint graph over background
    background = Image.open(intersection_fig + horizontal_light + '_' + vertical_light + '.png')
    x_lim, y_lim = background.size
    # update pedestrians
    for person in pedestrians:
        if (person.state[0] <= x_lim and person.state[0] >= 0 and person.state[1] >= 0 and person.state[1] <= y_lim):
            person.prim_next(dt)
            draw_pedestrian(person)
    # update planner
    # TODO: integrate planner
    # update enemy cars
    for vehicle in enemy_cars:
        nu = 0
        acc = 0
        if (vehicle.state[2] >= 0 and vehicle.state[3] >= 0 and vehicle.state[2] <= x_lim and vehicle.state[3] <= y_lim):
            if random.random() > 0.1:
                nu = random.uniform(-0.02,0.02)
            acc = random.uniform(-5,10)
            vehicle.next((acc, nu),dt)
            draw_car(vehicle)
    stage = plt.imshow(background, origin="lower") # this origin option flips the y-axis

    if i > delay_time:
        for vehicle in delayed_enemy_cars:
            nu = 0
            acc = 0
            if (vehicle.state[2] >= 0 and vehicle.state[3] >= 0 and vehicle.state[2] <= x_lim and vehicle.state[3] <= y_lim):
                if random.random() > 0.1:
                    nu = random.uniform(-0.02,0.02)
                acc = random.uniform(-5,10)
                vehicle.next((acc, nu),dt)
                draw_car(vehicle)
        stage = plt.imshow(background, origin="lower") # this origin option flips the y-axis

    if i <= delay_time:
        for vehicle in waiting_enemy_cars:
            nu = 0
            acc = 0
            vehicle.next((acc, nu),dt)
            draw_car(vehicle)
        stage = plt.imshow(background, origin="lower") # this origin option flips the y-axis

    ## update controlled cars with primitives
    for vehicle in controlled_cars:
        vehicle.prim_next(dt = dt)
        if vehicle.prim_queue.len() > 0:
            draw_car(vehicle)
    stage = plt.imshow(background, origin="lower") # this origin option flips the y-axis

#        dots = plt.axes().plot(240,300,'ro')
#        return stage, dots  # notice the comma is required to make returned object iterable (a requirement of FuncAnimation)
    return stage,   # notice the comma is required to make returned object iterable (a requirement of FuncAnimation)
##
## OBSERVER GOES HERE 
## TAKES IN CONTRACTS, CARS AND TRAFFIC LIGHT
##
t0 = time()
animate(0)
t1 = time()
interval = (t1 - t0)
show_waypoint_graph = False
save_video = False
num_frames = 550 # number of the first frames to save in video
ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=True,
        init_func = init, repeat=False) # by default the animation function loops, we set repeat to False in order to limit the number of frames generated to num_frames

if save_video:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('movies/hi_quality.avi', writer=writer, dpi=200)
plt.show()
