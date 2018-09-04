#!/usr/local/bin/python
# Visualization
# Tung M. Phan
# California Institute of Technology
# July 17, 2018

import os, platform, time, warnings, matplotlib, random
import components.planner as planner
import components.car as car
import components.aux.honk_wavefront as wavefront
import components.pedestrian as pedestrian
import components.traffic_signals as traffic_signals
import components.intersection as intersection
import prepare.car_waypoint_graph as car_graph
import prepare.pedestrian_waypoint_graph as pedestrian_graph
import prepare.graph as graph
import prepare.queue as queue
import assumes.params as params
import primitives.tubes as tubes
import primitives.load_primitives as load_primitives
from primitives.load_primitives import get_prim_data
from components.aux.pedestrian_names import names
import prepare.options as options
from  prepare.collision_check import collision_free, get_bounding_box
import numpy as np
from numpy import cos, sin, tan, pi
from PIL import Image
import scipy.io
if platform.system() == 'Darwin': # if the operating system is MacOS
    matplotlib.use('macosx')
else: # if the operating system is Linux or Windows
    try:
        import PySide2 # if pyside2 is installed
        matplotlib.use('Qt5Agg')
    except ImportError:
        warnings.warn('Using the TkAgg backend, this may affect performance. Consider installing pyside2 for Qt5Agg backend')
        matplotlib.use('TkAgg') # this may be slower
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# set dir_path to current directory
dir_path = os.path.dirname(os.path.realpath(__file__))
intersection_fig = dir_path + "/components/imglib/intersection_states/intersection_"

G = graph.WeightedDirectedGraph() # primitive graph
edge_to_prim_id = np.load('prepare/edge_to_prim_id.npy').item()

for prim_id in range(0, load_primitives.num_of_prims):
    try:
        controller_found = get_prim_data(prim_id, 'controller_found')[0]
        if controller_found:
            from_node = tuple(get_prim_data(prim_id, 'x0'))
            to_node = tuple(get_prim_data(prim_id, 'x_f'))
            time_weight = get_prim_data(prim_id, 't_end')[0]
            new_edge = (from_node, to_node, time_weight)
            new_edge_set = [new_edge] # convert to tuple otherwise, not hashable (can't check set membership)

            G.add_edges(new_edge_set, use_euclidean_weight=False)
            G.add_edges(new_edge_set, use_euclidean_weight=False)

            from_x = from_node[2]
            from_y = from_node[3]

            to_x = to_node[2]
            to_y = to_node[3]

            if (from_x, from_y) in car_graph.G._sources:
                G.add_source(from_node)
            if (to_x, to_y) in car_graph.G._sinks:
                G.add_sink(to_node)
    except ValueError:
        pass

def find_corner_coordinates(x_state_center_before, y_state_center_before, x_desired, y_desired, theta, square_fig):
    """
    This function takes an image and an angle then computes
    the coordinates of the corner (observe that vertical axis here is flipped).
    If we'd like to put the point specfied by (x_state_center_before, y_state_center_before) at (x_desired, y_desired),
    this function returns the coordinates of the lower left corner of the new image
    """
    w, h = square_fig.size
    theta = -theta
    if abs(w - h) > 1:
        warnings.warn("Warning: Figure has to be square! Otherwise, clipping or unexpected behavior may occur")
    R = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    x_corner_center_before, y_corner_center_before = -w/2., -h/2. # lower left corner before rotation
    x_corner_center_after, y_corner_center_after = -w/2., -h/2. # doesn't change since figure size remains unchanged

    x_state_center_after, y_state_center_after = R.dot(np.array([[x_state_center_before], [y_state_center_before]])) # relative coordinates after rotation by theta

    x_state_corner_after = x_state_center_after - x_corner_center_after
    y_state_corner_after = y_state_center_after - y_corner_center_after

    # x_corner_unknown + x_state_corner_after = x_desired
    x_corner_unknown = int(x_desired - x_state_center_after + x_corner_center_after)
    # y_corner_unknown + y_state_corner_after = y_desired
    y_corner_unknown = int(y_desired - y_state_center_after + y_corner_center_after)
    return x_corner_unknown, y_corner_unknown

def with_probability(P=1):
    return np.random.uniform() <= P

def draw_pedestrians(pedestrians):
    for pedestrian in pedestrians:
        x, y, theta, current_gait = pedestrian.state
        i = current_gait % pedestrian.film_dim[1]
        j = current_gait // pedestrian.film_dim[1]
        film_fig = Image.open(pedestrian.fig)
        scaled_film_fig_size  =  tuple([int(params.pedestrian_scale_factor * i) for i in film_fig.size])
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

# set random seed (for debugging)
# random.seed(99)
# np.random.seed(99)

# disable antialiasing for better performance
antialias_enabled = True
def draw_cars(vehicles):
    for vehicle in vehicles:
        vee, theta, x, y = vehicle.state
        # convert angle to degrees and positive counter-clockwise
        theta_d = -theta/np.pi * 180
        vehicle_fig = vehicle.fig
        w_orig, h_orig = vehicle_fig.size
        # set expand=True so as to disable cropping of output image
        vehicle_fig = vehicle_fig.rotate(theta_d, expand = False)
        scaled_vehicle_fig_size  =  tuple([int(params.car_scale_factor * i) for i in vehicle_fig.size])
        # rescale car 
        if antialias_enabled:
            vehicle_fig = vehicle_fig.resize(scaled_vehicle_fig_size, Image.ANTIALIAS)
        else:
            vehicle_fig = vehicle_fig.resize(scaled_vehicle_fig_size)
        # at (full scale) the relative coordinates of the center of the rear axle w.r.t. the center of the figure is center_to_axle_dist
        x_corner, y_corner = find_corner_coordinates(-params.car_scale_factor * params.center_to_axle_dist, 0, x, y, theta, vehicle_fig)
        background.paste(vehicle_fig, (x_corner, y_corner), vehicle_fig)

medic_signs = []
medic_sign = dir_path + '/components/imglib/medic.png'
medic_fig = Image.open(medic_sign).convert("RGBA")
medic_fig = medic_fig.resize((15,15))
def draw_medic_signs(medic_signs):
    for coordinate in medic_signs:
        x, y = find_corner_coordinates(0, 0, coordinate[0], coordinate[1], 0, medic_fig)
        background.paste(medic_fig, (int(x), int(y)), medic_fig)  

walk_sign_go = dir_path + '/components/imglib/go.png'
go_fig = Image.open(walk_sign_go).convert("RGBA")
go_fig = go_fig.resize((18,25))

walk_sign_stop = dir_path + '/components/imglib/stop.png'
stop_fig = Image.open(walk_sign_stop).convert("RGBA")
stop_fig = stop_fig.resize((18,25))

vertical_go_fig = go_fig.rotate(-180, expand = False)
vertical_stop_fig = stop_fig.rotate(-180, expand = False)
horizontal_go_fig = go_fig.rotate(-90, expand = True)
horizontal_stop_fig = stop_fig.rotate(-90, expand = True)

def draw_walk_sign(orientation, safe_to_walk):
    if orientation == 'vertical':
        if safe_to_walk:
            x, y = find_corner_coordinates(0, 0, 380, 670, 0, vertical_go_fig)
            background.paste(vertical_go_fig, (int(x), int(y)), vertical_go_fig)
        else:
            x, y = find_corner_coordinates(0, 0, 380, 670, 0, vertical_stop_fig)
            background.paste(vertical_stop_fig, (int(x), int(y)), vertical_stop_fig)
    elif orientation == 'horizontal':
        if safe_to_walk:
            x, y = find_corner_coordinates(0, 0, 260, 195, 0, horizontal_go_fig)
            background.paste(horizontal_go_fig, (int(x), int(y)), horizontal_go_fig)
        else:
            x, y = find_corner_coordinates(0, 0, 260, 195, 0, horizontal_stop_fig)
            background.paste(horizontal_stop_fig, (int(x), int(y)), horizontal_stop_fig)
    else:
        TypeError('Orientation must be vertical or horizontal')

# creates figure
fig = plt.figure()
ax = fig.add_axes([0,0,1,1]) # get rid of white border

# turn on/off axes
show_axes = False
if not show_axes:
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
    color = np.random.choice(['gray','blue'])
    the_car = car.KinematicCar(init_state=start_node, color=color, plate_number=plate_number)
    return plate_number, start_node, end_node, the_car

def spawn_pedestrian():
    name = random.choice(names)
    age = random.randint(18,70)
    start_node = random.sample(pedestrian_graph.G._sources, 1)[0]
    end_node = random.sample(pedestrian_graph.G._sinks, 1)[0]
    init_state = start_node + (0,0)
    if age > 50:
        pedestrian_type = '2'
    else:
        pedestrian_type = random.choice(['1','3'])
    the_pedestrian = pedestrian.Pedestrian(init_state=init_state, pedestrian_type=pedestrian_type, name=name, age=age)
    return name, start_node, end_node, the_pedestrian

def path_to_primitives(path):
    primitives = []
    for node_s, node_e  in zip(path[:-1], path[1:]):
            next_prim_id = edge_to_prim_id[(node_s, node_e)]
            primitives.append(next_prim_id)
    return primitives

#if true, pedestrians can cross street and cars cannot cross
def safe_to_walk(green_duration, light_color, light_time):
    return(light_color == 'green' and light_time < (green_duration / 3))

# create traffic lights
traffic_lights = traffic_signals.TrafficLights(3, 23, random_start = True)
#init_horizontal_light = traffic_lights._state['horizontal']
#init_vertical_light = traffic_lights._state['vertical']
#init_lights = {'horizontal': init_horizontal_light, 'vertical': init_vertical_light}
horizontal_light = traffic_lights.get_states('horizontal', 'color')
vertical_light = traffic_lights.get_states('vertical', 'color')
background = Image.open(intersection_fig + horizontal_light + '_' + vertical_light + '.png')

pedestrians = []
pedestrian_queue = queue.Queue()

cars = dict()
edge_time_stamps = dict()
time_stamps = dict()
request_queue = queue.Queue()
honk_x = []
honk_y = []
honk_t = []
all_wavefronts = set()
request_queue = queue.Queue()
waiting = dict()

def pause_car(the_car):
    the_car.prim_queue.enqueue((-1, 0))

def unpause_car(the_car, plate_number):
    the_car.prim_queue.remove((-1,0))
    if plate_number in waiting.keys():
        del waiting[plate_number]

def clean_stamps(edge_time_stamps, current_time):
    for key in edge_time_stamps.copy():
        for interval in edge_time_stamps[key].copy():
            if interval[1] < current_time:
                edge_time_stamps[key].remove(interval)
            if len(edge_time_stamps[key]) == 0:
                del edge_time_stamps[key]

effective_current_times = dict()

# checks if pedestrian is crossing street
def distance(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
def is_between(lane, person_xy):
    return distance(lane[0], person_xy) + distance(lane[1], person_xy) == distance(lane[0], lane[1])

# cross the street if light is green, or if the pedestrian just crossed the street, continue walking and ignore traffic light color
def continue_walking(person, walk_sign, lane1, lane2, direction):
    person_xy = (person.state[0], person.state[1])
    theta = person.state[2]
    if person_xy in (lane1 + lane2) and walk_sign:
        return True
    elif (person_xy == lane1[0] or person_xy == lane2[0]) and theta == direction[0]:
        return True
    elif (person_xy == lane1[1] or person_xy == lane2[1]) and theta == direction[1]:
        return True
    else:
        return False

# if the pedestrian isn't going fast enough, increase speed to cross street before light turns red
def walk_faster(person, remaining_time):
    person_xy = (person.state[0], person.state[1])
    prim_data, prim_progress = person.prim_queue.top()
    start, finish, vee = prim_data
    dx = finish[0] - person.state[0]
    dy = finish[1] - person.state[1]
    remaining_distance = np.linalg.norm(np.array([dx, dy]))
    if (remaining_distance / vee) > remaining_time:
        vee = remaining_distance / remaining_time
        person.prim_queue.replace_top(((start, finish, vee), prim_progress))

def animate(frame_idx): # update animation by dt
    deadlock = False
    current_time = frame_idx * dt # compute current time from frame index and dt
    print('{:.2f}'.format(current_time)) # print out current time to 2 decimal places

    horizontal_light = traffic_lights.get_states('horizontal', 'color')
    vertical_light = traffic_lights.get_states('vertical', 'color')

    horizontal_light_time = traffic_lights.get_states('horizontal', 'time')
    vertical_light_time = traffic_lights.get_states('vertical', 'time')
    green_duration = traffic_lights._max_time['green']

    # if sign is true then walk, stop if false
    vertical_walk_safe = safe_to_walk(green_duration, vertical_light, vertical_light_time)
    horizontal_walk_safe = safe_to_walk(green_duration, horizontal_light, horizontal_light_time)
    draw_walk_sign('vertical', vertical_walk_safe)
    draw_walk_sign('horizontal', horizontal_walk_safe)

    """ online frame update """
    global background
    if with_probability(0.2):
#    if with_probability(min(1,10/(frame_idx+1))):
        new_plate_number, new_start_node, new_end_node, new_car = spawn_car()
        request_queue.enqueue((new_plate_number, new_start_node, new_end_node, new_car))
    service_count = 0
    original_request_len = request_queue.len()
    while request_queue.len() > 0 and not deadlock: # if there is at least one request in the queue
        plate_number, prestart_node, end_node, the_car = request_queue.pop() # take the first request
#        ii = 60
#        for jj in range(params.num_subprims):
#            if (ii,jj) in edge_time_stamps:
#                print('time stamp for (' + str(ii) +  ',' + str(jj) + ')' + str(edge_time_stamps[(ii,jj)]))
#        ii = 62
#        for jj in range(params.num_subprims):
#            if (ii,jj) in edge_time_stamps:
#                print('time stamp for (' + str(ii) +  ',' + str(jj) + ')' + str(edge_time_stamps[(ii,jj)]))
        start_velocity = prestart_node[0]
        if plate_number not in effective_current_times.keys():
            effective_current_times[plate_number] = current_time
        else:
            effective_current_times[plate_number] = max(current_time, effective_current_times[plate_number])
        if the_car.prim_queue.len() > 1:
            if the_car.prim_queue.top()[0] == -1:
                start_velocity = 0
        start_node = (start_velocity, prestart_node[1], prestart_node[2], prestart_node[3])
        service_count += 1
        # consider the case where the plate number is already in the waiting set
        if plate_number in waiting.keys(): # if plate number is already waiting
            wait_id, interval = waiting[plate_number]
            edge_time_stamps[wait_id].remove(interval)
        _, shortest_path = planner.dijkstra(start_node, end_node, G)

        safety_check, first_conflict_edge_idx = planner.is_safe(path = shortest_path, current_time = effective_current_times[plate_number], primitive_graph = G, edge_time_stamps = edge_time_stamps, traffic_lights = traffic_lights, walk_signs = (vertical_walk_safe, horizontal_walk_safe))
        if safety_check:
            if plate_number not in cars:
                cars[plate_number] = the_car # add the car
            if plate_number in waiting.keys():
                edge_time_stamps[wait_id].add((interval[0], effective_current_times[plate_number]))
            unpause_car(the_car, plate_number)
            planner.time_stamp_edge(path = shortest_path, edge_time_stamps = edge_time_stamps, current_time = effective_current_times[plate_number], primitive_graph = G)
            path_prims = path_to_primitives(shortest_path) # add primitives
            for prim_id in path_prims:
                cars[plate_number].prim_queue.enqueue((prim_id, 0))
                prim_time = get_prim_data(prim_id, 't_end')[0]
                effective_current_times[plate_number] += prim_time
        elif first_conflict_edge_idx == None:
            if plate_number in waiting.keys():
                edge_time_stamps[wait_id].add(interval) # add temporarily removed interval back
            new_request = (plate_number, start_node, end_node, the_car)
            request_queue.enqueue(new_request)
        else:
            if first_conflict_edge_idx == 0 and plate_number not in cars:
                cars[plate_number] = the_car # add the car
                pause_car(the_car)
                prim_id = path_to_primitives(shortest_path[:2])[0]
                wait_interval = (effective_current_times[plate_number], float('inf'))
                wait_id = (prim_id, 0)
                waiting[plate_number] = (wait_id, wait_interval) # add/update current node there
                try:
                    edge_time_stamps[wait_id].add(wait_interval)
                except KeyError:
                    edge_time_stamps[wait_id] = {wait_interval}
            elif first_conflict_edge_idx == 0 and plate_number in cars:
                if plate_number in waiting:
                    wait_id, wait_interval = waiting[plate_number]
                    edge_time_stamps[wait_id].add(wait_interval) # add temporarily removed interval back
            else: # if first_conflict_edge_idx > 0
                unpause_car(the_car, plate_number)
                if plate_number not in cars:
                    cars[plate_number] = the_car # add the car
                if plate_number in waiting.keys(): # if plate number is already waiting
                    edge_time_stamps[wait_id].add((interval[0], effective_current_times[plate_number]))
                partial_path = shortest_path[:first_conflict_edge_idx+1]
                _, last_interval = planner.time_stamp_edge(path = partial_path, edge_time_stamps = edge_time_stamps, current_time = effective_current_times[plate_number], primitive_graph = G, partial=True)
                path_prims = path_to_primitives(path=partial_path) # add primitives
                for prim_id in path_prims:
                    cars[plate_number].prim_queue.enqueue((prim_id, 0))
                    prim_time = get_prim_data(prim_id, 't_end')[0]
                    effective_current_times[plate_number] += prim_time
                pause_car(the_car)
                start_node = (partial_path[-1][0], partial_path[-1][1], partial_path[-1][2], partial_path[-1][3])
                wait_id = (path_prims[-1], params.num_subprims-1)
                waiting[plate_number] = (wait_id, last_interval) # add/update current node there
            new_request = (plate_number, start_node, end_node, the_car)
            request_queue.enqueue(new_request)
        if service_count == original_request_len:
            service_count = 0 # reset service count
            if request_queue.len() == original_request_len:
#                print('deadlock!!')
                deadlock = True
            else:
                original_request_len = request_queue.len()

######## pedestrian implementation ########
    if with_probability(0.05):
        new_name, new_begin_node, new_final_node, new_pedestrian = spawn_pedestrian()
        if new_begin_node == new_final_node:
            print("Request Denied")
        else:
            pedestrian_queue.enqueue((new_name, new_begin_node, new_final_node, new_pedestrian))
    while pedestrian_queue.len() > 0: 
        name, begin_node, final_node, the_pedestrian = pedestrian_queue.pop()
        _, shortest_path = planner.dijkstra((begin_node[0], begin_node[1]), final_node, pedestrian_graph.G)
        vee = np.random.uniform(20, 40)
        while len(shortest_path) > 0:
            if len(shortest_path) == 1:
                the_pedestrian.prim_queue.enqueue(((shortest_path[0], shortest_path[0], vee), 0))
                del shortest_path[0]
            else:
                the_pedestrian.prim_queue.enqueue(((shortest_path[0], shortest_path[1], vee), 0))
                del shortest_path[0]
        pedestrians.append(the_pedestrian)
######## pedestrian implementation ########

    # update traffic lights
    traffic_lights.update(dt)
    horizontal_light = traffic_lights.get_states('horizontal', 'color')
    vertical_light = traffic_lights.get_states('vertical', 'color')

    # update background
    background.close()
    background = Image.open(intersection_fig + horizontal_light + '_' + vertical_light + '.png')
    x_lim, y_lim = background.size

    # if sign is true then walk, stop if false
    vertical_walk_safe = safe_to_walk(green_duration, vertical_light, vertical_light_time)
    horizontal_walk_safe = safe_to_walk(green_duration, horizontal_light, horizontal_light_time)
    draw_walk_sign('vertical', vertical_walk_safe)
    draw_walk_sign('horizontal', horizontal_walk_safe)

    # update pedestrians
    pedestrians_to_keep = []
    pedestrians_waiting = []
    lane1 = [(355, 195), (355, 565)] # left vertical path, (bottom node, top node)
    lane2 = [(705, 195), (705, 565)] # right vertical path, (bottom node, top node)
    lane3 = [(380, 590), (680, 590)] # top horizontal path (left node, right node)
    lane4 = [(380, 170), (680, 170)] # bottom horizontal path (left node, right node)

    if len(pedestrians) > 0:
        for person in pedestrians:
            if (person.state[0] <= x_lim and person.state[0] >= 0 and person.state[1] >= 0 and person.state[1] <= y_lim):
                person_xy = (person.state[0], person.state[1])
                if person_xy not in (lane1 + lane2 + lane3 + lane4): # if pedestrian is not at any of the nodes then continue  
                    person.prim_next(dt)
                    pedestrians_to_keep.append(person)
                elif continue_walking(person, vertical_walk_safe, lane1, lane2, (-pi/2, pi/2)): # if light is green cross the street, or if at a node and facing away from the street i.e. just crossed the street then continue
                    person.prim_next(dt)
                    pedestrians_to_keep.append(person)
                elif continue_walking(person, horizontal_walk_safe, lane3, lane4, (pi, 0)):
                    person.prim_next(dt)
                    pedestrians_to_keep.append(person)
                else:
                    person.state[3] = 0
                    pedestrians_waiting.append(person)

                # pedestrians walk faster if not going fast enough to finish crossing the street before walk sign is off or 'false'
                walk_sign_duration = green_duration / 3
                if is_between(lane1, person_xy) or is_between(lane2, person_xy):
                    remaining_vertical_time = abs(walk_sign_duration - vertical_light_time)
                    walk_faster(person, remaining_vertical_time)
                elif is_between(lane3, person_xy) or is_between(lane4, person_xy):
                    remaining_horizontal_time = abs(walk_sign_duration  - horizontal_light_time)
                    walk_faster(person, remaining_horizontal_time)

    cars_to_keep = []
    cars_to_remove = set()

    # determine which cars to keep
    for plate_number in cars.keys():
        car = cars[plate_number]
#        if (car.state[2] <= x_lim and car.state[2] >= 0 and car.state[3] >= 0 and car.state[3] <= y_lim): TODO: implement this
        if car.prim_queue.len() > 0:
            # update cars with primitives
            cars[plate_number].prim_next(dt)
            # add cars to keep list
            cars_to_keep.append(cars[plate_number])
        else:
            cars_to_remove.add(plate_number)
    # determine which cars to remove
    for plate_number in cars_to_remove:
        del cars[plate_number]

    ax.cla() # clear Axes before plotting
    clean_stamps(edge_time_stamps, current_time)
    if not show_axes:
        plt.axis('off')
    ## STAGE UPDATE HAPPENS AFTER THIS COMMENT
    honk_waves = ax.scatter(None,None)
    if options.show_honks:
        honk_xs = []
        honk_ys = []
        radii = []
        intensities = []
        for wave in all_wavefronts:
            wave.next(dt)
            honk_x, honk_y, radius, intensity = wave.get_data()
            if intensity > 0:
                honk_xs.append(honk_x)
                honk_ys.append(honk_y)
                radii.append(radius)
                intensities.append(intensity)
            else:
                all_wavefronts.remove(wave)

        rgba_colors = np.zeros((len(intensities),4))
        # red color
        rgba_colors[:,0] = 1.0
        # intensities
        rgba_colors[:, 3] = intensities
        honk_waves = ax.scatter(honk_xs, honk_ys, s = radii, lw=1, facecolors='none', color=rgba_colors)

    for car in cars_to_keep:
        if with_probability(0.005) and not car.is_honking:
            car.toggle_honk()
            # offset is 600 before scaling
            wave = wavefront.HonkWavefront([car.state[2] + 600*params.car_scale_factor*np.cos(car.state[1]), car.state[3] + 600*params.car_scale_factor*np.sin(car.state[1]), 0, 0], init_energy=100000)
            all_wavefronts.add(wave)
        elif with_probability(0.4) and car.is_honking:
            car.toggle_honk()

    # initialize boxes
    boxes = [ax.plot([], [], 'c')[0] for _ in range(len(cars_to_keep))]

    if options.show_boxes:
        for i in range(len(cars_to_keep)):
            curr_car = cars_to_keep[i]
            vertex_set,_,_,_ = get_bounding_box(curr_car)
            xs = [vertex[0] for vertex in vertex_set]
            ys = [vertex[1] for vertex in vertex_set]
            xs.append(vertex_set[0][0])
            ys.append(vertex_set[0][1])
            if with_probability(1):
               boxes[i].set_data(xs,ys)
            for j in range(i + 1, len(cars_to_keep)):
               if not collision_free(curr_car, cars_to_keep[j]):
                    boxes[j].set_color('r')
                    boxes[i].set_color('r')
    # initialize ids
    ids = [ax.text([], [], '') for _ in range(len(cars_to_keep))]
    if options.show_ids:
        for i in range(len(cars_to_keep)):
            _,_,x,y = cars_to_keep[i].state
            plate_number = cars_to_keep[i].plate_number
            ids[i] = ax.text(x,y,str(plate_number), color='w', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='red', alpha=0.5), fontsize=10)

    # initialize prims
    plot_prims = [ax.text([], [], '') for _ in range(len(cars_to_keep))]
    if options.show_prims:
        for i in range(len(cars_to_keep)):
            _,_,x,y = cars_to_keep[i].state
            prim_str = 'None'
            if cars_to_keep[i].prim_queue.len() > 0:
                prim_str = str((cars_to_keep[i].prim_queue.top()[0], int(params.num_subprims*cars_to_keep[i].prim_queue.top()[1])))
            plot_prims[i] = ax.text(x,y, prim_str, color='w', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='red', alpha=0.5), fontsize=10)

    # plot primitive tubes
    curr_tubes = []
    # initialize tubes
    if options.show_tubes:
        curr_tubes = [ax.plot([], [],'b')[0] for _ in range(len(cars_to_keep))]

        for i in range(len(cars_to_keep)):
            curr_car = cars_to_keep[i]
            if curr_car.prim_queue.len() > 0:
                if curr_car.prim_queue.top()[1] < 1:
                    curr_prim_id = curr_car.prim_queue.top()[0]
                    if curr_prim_id == -1:
                        pass
                    else:
                        curr_prim_progress = curr_car.prim_queue.top()[1]
                        vertex_set = tubes.make_tube(curr_prim_id)
                        xs = [vertex[0][0] for vertex in vertex_set[int(curr_prim_progress * params.num_subprims )]]
                        ys = [vertex[1][0] for vertex in vertex_set[int(curr_prim_progress * params.num_subprims )]]
                        xs.append(xs[0])
                        ys.append(ys[0])
                        curr_tubes[i].set_data(xs,ys)

    # if cars are hitting pedestrians replace pedestrian with medic sign
    for person in pedestrians_to_keep:
        for car in cars_to_keep:
            collision_free1,_ = collision_free(person, car)
            if not collision_free1 and car.state[0] > 1:
                medic_signs.append((person.state[0], person.state[1]))
                del pedestrians[pedestrians.index(person)]

    # plot honking 
    np.random.shuffle(pedestrians_waiting)
    np.random.shuffle(pedestrians_to_keep)
    draw_pedestrians(pedestrians_waiting)
    draw_medic_signs(medic_signs)
    draw_pedestrians(pedestrians_to_keep) # draw pedestrians to background
    draw_cars(cars_to_keep)


    # plot traffic light walls
    walls = [ax.plot([], [],'r')[0] for _ in range(4)]
    if options.show_traffic_light_walls:
        if traffic_lights.get_states('horizontal', 'color') == 'red':
            xs = intersection.traffic_light_walls['west']['x']
            ys = intersection.traffic_light_walls['west']['y']
            xs.append(xs[0])
            ys.append(ys[0])
            walls[0].set_data(xs,ys)
            xs = intersection.traffic_light_walls['east']['x']
            ys = intersection.traffic_light_walls['east']['y']
            xs.append(xs[0])
            ys.append(ys[0])
            walls[1].set_data(xs,ys)
        elif traffic_lights.get_states('vertical', 'color') == 'red':
            xs = intersection.traffic_light_walls['north']['x']
            ys = intersection.traffic_light_walls['north']['y']
            xs.append(xs[0])
            ys.append(ys[0])
            walls[2].set_data(xs,ys)
            xs = intersection.traffic_light_walls['south']['x']
            ys = intersection.traffic_light_walls['south']['y']
            xs.append(xs[0])
            ys.append(ys[0])
            walls[3].set_data(xs,ys)



    # show artists
    global stage # set up a global stage
    stage = ax.imshow(background, origin="lower") # update the stage
    return  [stage] + [honk_waves] + boxes + curr_tubes + ids + plot_prims + walls

t0 = time.time()
animate(0)
t1 = time.time()
interval = (t1 - t0)
num_frames = 2000 # number of the first frames to save in video
ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=True, repeat=False) # by default the animation function loops, we set repeat to False in order to limit the number of frames generated to num_frames

if options.save_video:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps = 30, metadata=dict(artist='Me'), bitrate=-1)
    ani.save('movies/peds.avi', writer=writer, dpi=200)
plt.show()
t2 = time.time()
print('Total elapsed time: ' + str(t2-t0))
