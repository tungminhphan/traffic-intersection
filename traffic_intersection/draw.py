#!/usr/local/bin/python
# Visualization
# Tung M. Phan
# California Institute of Technology
# July 17, 2018

import sys, os
sys.path.append('..')
import traffic_intersection.components.planner as planner
import traffic_intersection.components.car as car
import traffic_intersection.components.pedestrian as pedestrian
import traffic_intersection.components.traffic_signals as traffic_signals
from traffic_intersection.prepare.collision_check import collision_check
import traffic_intersection.prepare.car_waypoint_graph as car_graph
import traffic_intersection.prepare.graph as graph
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from time import time
from numpy import cos, sin, tan
import numpy as np
from PIL import Image
import random
import scipy.io

# set dir_path to current directory
dir_path = os.path.dirname(os.path.realpath(__file__))
intersection_fig = dir_path + "/components/imglib/intersection_states/intersection_"
car_scale_factor = 0.1 # scale for when L = 50
pedestrian_scale_factor = 0.32
# load primitive data
primitive_data = dir_path + '/primitives/MA3.mat'
mat = scipy.io.loadmat(primitive_data)
num_of_prims = mat['MA3'].shape[0]

def get_prim_data(prim_id, data_field):
    '''
    This function simplifies the process of extracting data from the .mat file
    Input:
    prim_id: the index of the primitive the data of which we would like to return
    data_field: name of the data field (e.g., x0, x_f, controller_found etc.)

    Output: the requested data
    '''
    return mat['MA3'][prim_id,0][data_field][0,0][:,0]


G = graph.WeightedDirectedGraph()
edge_to_prim_id = dict() # dictionary to convert primitive move to primitive ID
prim_id_to_edge = dict() # dictionary to convert primitive ID to edge

for prim_id in range(0, num_of_prims):
    if get_prim_data(prim_id, 'controller_found')[0] == 1:
        from_node = tuple(get_prim_data(prim_id, 'x0'))
        to_node = tuple(get_prim_data(prim_id, 'x_f'))
        new_edge = (from_node, to_node)
        edge_to_prim_id[new_edge] = prim_id
        prim_id_to_edge[prim_id] = new_edge

        new_edge_set = [new_edge] # convert to tuple otherwise, not hashable (can't check set membership)
        G.add_edges(new_edge_set)

        from_x = from_node[2]
        from_y = from_node[3]

        to_x = to_node[2]
        to_y = to_node[3]

        if (from_x, from_y) in car_graph.G._sources:
            G.add_source(from_node)
        if (to_x, to_y) in car_graph.G._sinks:
            G.add_sink(to_node)

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
# create car
def spawn_car():
    def generate_license_plate():
        import string
        choices = string.digits + string.ascii_uppercase
        plate_number = ''
        for i in range(0,7):
            plate_number = plate_number + random.choice(choices)
        return plate_number
    plate_number = generate_license_plate()
    rand_num = np.random.choice(10)
    start_node = random.sample(G._sources, 1)[0]
    end_node = random.sample(G._sinks, 1)[0]
    shortest_path_length, shortest_path = planner.dijkstra(start_node, end_node, G)

    the_car = car.KinematicCar(init_state=start_node, color='gray')
    for node_s, node_e  in zip(shortest_path[:-1], shortest_path[1:]):
            next_prim_id = edge_to_prim_id[(node_s, node_e)]
            the_car.prim_queue.enqueue((next_prim_id, 0))

    return plate_number, the_car

# create traffic lights
traffic_lights = traffic_signals.TrafficLights(3, 23, random_start = False)
horizontal_light = traffic_lights.get_states('horizontal', 'color')
vertical_light = traffic_lights.get_states('vertical', 'color')
background = Image.open(intersection_fig + horizontal_light + '_' + vertical_light + '.png')
def init():
    stage = plt.imshow(background, origin="lower",zorder=0) # this origin option flips the y-axis
    return stage, # notice the comma is required to make returned object iterable (a requirement of FuncAnimation)

pedestrian_2 = pedestrian.Pedestrian(init_state=[500,500, np.pi/2, 2], pedestrian_type='2')
pedestrians = [pedestrian_2]
cars = dict()

def animate(i): # update animation by dt
    print(i)
    """ online frame update """
    global background
    if np.random.uniform() <= 0.03:
        plate_number, the_car = spawn_car()
        cars[plate_number] = the_car
    # update traffic lights
    traffic_lights.update(dt)
    horizontal_light = traffic_lights.get_states('horizontal', 'color')
    vertical_light = traffic_lights.get_states('vertical', 'color')
    # update background
    background = Image.open(intersection_fig + horizontal_light + '_' + vertical_light + '.png')
    x_lim, y_lim = background.size

    # update pedestrians
    if len(pedestrians) > 0:
        for person in pedestrians:
            if (person.state[0] <= x_lim and person.state[0] >= 0 and person.state[1] >= 0 and person.state[1] <= y_lim):
                draw_pedestrian(person)
        for i in range(len(pedestrians)):
            for j in range(i + 1, len(pedestrians)):
                if collision_check(pedestrians[i], pedestrians[j], car_scale_factor, pedestrian_scale_factor):
                    print('Collision between pedestrian' + str(i) + 'and '+  str(j))
                else:
                    print("No Collision")

    # update cars with primitives
    cars_to_remove = set()
    for plate_number in cars.keys():
        if cars[plate_number].prim_queue.len() > 0: # TODO: update this
            cars[plate_number].prim_next(dt)
            draw_car(cars[plate_number])
        else:
            cars_to_remove.add(plate_number)
    for plate_number in cars_to_remove:
        del cars[plate_number]
    stage = plt.imshow(background, origin="lower") # update the stage; the origin option flips the y-axis
    return stage,   # notice the comma is required to make returned object iterable (a requirement of FuncAnimation)

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
