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
import traffic_intersection.prepare.waypoint_graph as waypoint_graph
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
blue_car_fig = dir_path + "/components/imglib/cars/blue_car.png"
gray_car_fig = dir_path + "/components/imglib/cars/gray_car.png"
car_scale_factor = 0.12
pedestrian_scale_factor = 0.6

def find_corner_coordinates(x_rel_i, y_rel_i, x_des, y_des, theta, square_fig):
    """
    This function takes an image and an angle then computes
    the coordinates of the corner (observe that vertical axis here is flipped)
    """
    w, h = square_fig.size
    theta = -theta
    if w != h:
        raise Exception("Figure has to be square!")
    x_corner_rel, y_corner_rel = -w/2, -h/2
    R = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    x_rel_f, y_rel_f = R.dot(np.array([[x_rel_i], [y_rel_i]]))
    # xy_unknown - xy_corner + xy_rel_f = xy_des
    return int(x_des - x_rel_f + x_corner_rel), int(y_des - y_rel_f + y_corner_rel)

def draw_car(vehicle):
    vee, theta, x, y = vehicle.state
    # convert angle to degrees and positive counter-clockwise
    theta_d = -theta/np.pi * 180
    if vehicle.color  == 'blue':
        vehicle_fig = Image.open(blue_car_fig)
    elif vehicle.color  == 'gray':
        vehicle_fig = Image.open(gray_car_fig)
    # set expand=True so as to disable cropping of output image
    vehicle_fig = vehicle_fig.rotate(theta_d, expand = False)
    scaled_vehicle_fig_size  =  tuple([int(car_scale_factor * i) for i in vehicle_fig.size])
    # rescale car 
    #vehicle_fig = vehicle_fig.resize(scaled_vehicle_fig_size, Image.ANTIALIAS)
    vehicle_fig = vehicle_fig.resize(scaled_vehicle_fig_size) # disable antialiasing for better performance
    # at this scale (-28, 0) is the relative coordinates of the center of the rear axle w.r.t. the
    # center of the figure
    x_corner, y_corner = find_corner_coordinates(-28, 0, x, y, theta, vehicle_fig)
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
    person_fig = person_fig.rotate(theta/np.pi * 180 + 90, expand = False)
    background.paste(person_fig, (int(x), int(y)), person_fig) 

# creates figure
fig = plt.figure()
fig.add_axes([0,0,1,1]) # get rid of white border

# turn on/off axes
plt.axis('off')
# sampling time
dt = 0.2
# Artist Animation option is used to generate offline movies - implemented here as a backup
use_artist_animation = False
if use_artist_animation:
    frames = []
    for i in range(0,100):
        # create new background
        background = Image.open(intersection_fig)
        # test values - should be changed according to needs
        car_1 = car.KinematicCar(init_state=(2,np.pi,300,300), L=50)
        draw_car(car_1)
        car_1.next((10, 0.001),dt)
        frames.append([plt.imshow(background, animated = True)])
    ani = animation.ArtistAnimation(fig, frames, interval = 10, repeat_delay=1)
    plt.show()
else:

    # creates cars
    prim_id = 16 # first primitive
    prim = mat['MA3'][prim_id,0]
    x0 = prim['x0'][0,0]
    car_1 = car.KinematicCar(init_state = np.reshape(x0, (-1, 1))) # primitive car
    car_1.prim_queue.enqueue((prim_id, 0))
#    car_1 = car.KinematicCar(init_state=(100,np.pi,1000,500)) # original test
    car_2 = car.KinematicCar(init_state=(20,np.pi/2,600,300), color='gray')
    car_3 = car.KinematicCar(init_state=(80,0,0,250), color='gray')
    car_4 = car.KinematicCar(init_state=(50,-np.pi/2,450,710), color='gray')
    enemy_cars = [car_2, car_3, car_4]
    controlled_cars = [car_1]
    # creates pedestrians
    pedestrian_1 = pedestrian.Pedestrian(init_state=[330,550,-np.pi/2,0])
    pedestrian_2 = pedestrian.Pedestrian(init_state=[680,0, np.pi/2,0])
    pedestrians = [pedestrian_1, pedestrian_2]
    # create traffic lights
    traffic_lights = traffic_signals.TrafficLights(3, 10)
    horizontal_light = traffic_lights.get_states('horizontal', 'color')
    vertical_light = traffic_lights.get_states('vertical', 'color')
    background = Image.open(intersection_fig + horizontal_light + '_' + vertical_light + '.png')
    def init():
        stage = plt.imshow(background, origin="lower",zorder=0) # this origin option flips the y-axis
        return stage, # notice the comma is required to make returned object iterable (a requirement of FuncAnimation)

    def animate(i): # update animation by dt
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
        if waypoint_graph == True:
            graph = waypoint_graph.plot_edges(plt, waypoint_graph.G, plt_src_snk=True)
            background.paste(graph, (0, 0), graph)
        # update pedestrians
        for person in pedestrians:
            dee_theta = 0
            v = 20
            if (person.state[0] <= x_lim and person.state[1] <= y_lim):
                person.next((dee_theta, v),dt)
                draw_pedestrian(person)
        # update planner
        # TODO: integrate planner
        # update enemy cars
        for vehicle in enemy_cars:
            nu = 0
            acc = 0
            if (vehicle.state[2] <= x_lim and vehicle.state[3] <= y_lim):
                if random.random() > 0.1:
                    nu = random.uniform(-0.05,0.05)
                if random.random() > 0.3:
                    acc = random.uniform(-5,10)
                vehicle.next((acc, nu),dt)
                draw_car(vehicle)
        stage = plt.imshow(background, origin="lower") # this origin option flips the y-axis

        ## update controlled cars with primitives
        for vehicle in controlled_cars:
            vehicle.prim_next(dt = dt)
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
    frames = 200 # number of the first frames to save in video
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True, init_func = init)

    if save_video:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('movies/new.avi', writer=writer, dpi=100)
plt.show()
