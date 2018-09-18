# Helper Functions
# Tung M. Phan
# September 14, 2018
import numpy as np
import random
from numpy import cos, sin, tan, pi
from PIL import Image, ImageDraw
import assumes.params as params
import prepare.options as options
import variables.global_vars as global_vars
import primitives.tubes as tubes
import prepare.car_waypoint_graph as car_graph
import components.car as car
import components.intersection as intersection
from  prepare.collision_check import collision_free, get_bounding_box
import components.auxiliary.honk_wavefront as wavefront
from components.auxiliary.pedestrian_names import names
import prepare.pedestrian_waypoint_graph as pedestrian_graph
import components.pedestrian as pedestrian
import components.traffic_signals as traffic_signals

def alt_sin(max_val,min_val,omega, x):
    A = 0.5*(max_val - min_val)
    b = 0.5*(max_val + min_val)
    return A * np.sin(omega*x) + b

def generate_walking_gif(): #TODO: check save directory
    pedestrian_fig = dir_path + "/imglib/pedestrians/walking3.png"
    film_dim = (1, 6) # dimension of the film used for animation
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
        print('Warning: Figure has to be square! Otherwise, clipping or unexpected behavior may occur')
#        warnings.warn("Warning: Figure has to be square! Otherwise, clipping or unexpected behavior may occur")


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

def draw_pedestrians(ax):
    global_vars.pedestrians_to_show = []
    for pedestrian in global_vars.pedestrians_to_keep:
        if not pedestrian.is_dead:
            x, y, theta, current_gait = pedestrian.state
            i = current_gait % pedestrian.film_dim[1]
            j = current_gait // pedestrian.film_dim[1]
            film_fig = Image.open(pedestrian.fig)
            scaled_film_fig_size  =  tuple([int(params.pedestrian_scale_factor * i) for i in film_fig.size])
            film_fig = film_fig.resize(scaled_film_fig_size)
            width, height = film_fig.size
            sub_width = width/pedestrian.film_dim[1]
            sub_height = height/pedestrian.film_dim[0]
            lower = (i*sub_width,j*sub_height)
            upper = ((i+1)*sub_width, (j+1)*sub_height)
            area = (int(lower[0]), int(lower[1]), int(upper[0]), int(upper[1]))
            person_fig = film_fig.crop(area)
            person_fig = person_fig.rotate(180+theta/np.pi * 180 + 90, expand = False)
            x_corner, y_corner = find_corner_coordinates(0., 0, x, y, theta,  person_fig)
        else:
            x, y, theta, _ = pedestrian.state
            person_fig = Image.open(pedestrian.fig)
            person_fig = person_fig.resize((20,20))
            x_corner, y_corner = find_corner_coordinates(0., 0., x, y, theta,  person_fig)
        w, h = person_fig.size
        global_vars.pedestrians_to_show.append(ax.imshow(person_fig, extent=(x_corner,x_corner+w, y_corner, y_corner+h)))
    random.shuffle(global_vars.pedestrians_to_show)


def spawn_car():
    plate_number = generate_license_plate()
    rand_num = np.random.choice(10)
    start_node = random.sample(car_graph.G._sources, 1)[0]
    end_node = random.sample(car_graph.G._sinks, 1)[0]
    color = np.random.choice(tuple(car.car_colors))
    the_car = car.KinematicCar(init_state=start_node, color=color, plate_number=plate_number)
    return start_node, end_node, the_car

def draw_cars(vehicles, background):
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
        if options.antialias_enabled:
            vehicle_fig = vehicle_fig.resize(scaled_vehicle_fig_size, Image.ANTIALIAS)
        else:
            vehicle_fig = vehicle_fig.resize(scaled_vehicle_fig_size)
        # at (full scale) the relative coordinates of the center of the rear axle w.r.t. the center of the figure is center_to_axle_dist
        x_corner, y_corner = find_corner_coordinates(-params.car_scale_factor * params.center_to_axle_dist, 0, x, y, theta, vehicle_fig)
        background.paste(vehicle_fig, (x_corner, y_corner), vehicle_fig)

def draw_cars_fast(ax, vehicles):
    global_vars.cars_to_show = []
    for vehicle in vehicles:
        vee, theta, x, y = vehicle.state
        # convert angle to degrees and positive counter-clockwise
        theta_d = theta/np.pi * 180
        vehicle_fig = vehicle.fig
        w_orig, h_orig = vehicle_fig.size
        # set expand=True so as to disable cropping of output image
        vehicle_fig = vehicle_fig.rotate(theta_d, expand = False)
        scaled_vehicle_fig_size  =  tuple([int(params.car_scale_factor * i) for i in vehicle_fig.size])
        # rescale car 
        if options.antialias_enabled:
            vehicle_fig = vehicle_fig.resize(scaled_vehicle_fig_size, Image.ANTIALIAS)
        else:
            vehicle_fig = vehicle_fig.resize(scaled_vehicle_fig_size)
        # at (full scale) the relative coordinates of the center of the rear axle w.r.t. the center of the figure is center_to_axle_dist
        x_corner, y_corner = find_corner_coordinates(-params.car_scale_factor*params.center_to_axle_dist, 0, x, y, theta, vehicle_fig)
        w, h = vehicle_fig.size
        global_vars.cars_to_show.append(ax.imshow(vehicle_fig, extent=(x_corner,x_corner+w, y_corner, y_corner+h)))

def with_probability(P=1):
    return np.random.uniform() <= P

def distance(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def generate_license_plate():
    import string
    choices = string.digits + string.ascii_uppercase
    plate_number = ''
    for i in range(0,7):
        plate_number = plate_number + random.choice(choices)
    return plate_number

def dijkstra(start, end, graph):
    '''
    this function takes in a weighted directed graph, a start node, an end node and outputs
    the shortest path from the start node to the end node on that graph
    input:  start - start node
            end - end node
            graph - weighted directed graph
    output: shortest path from start to end node
    '''
    if start == end: # if start coincides with end
        return 0, [start]
    else: # otherwise
        score = {}
        predecessor = {}
        unmarked_nodes = graph._nodes.copy() # create a copy of set of nodes in graph
        if start not in graph._nodes:
            print(start)
            raise SyntaxError("The start node is not in the graph!")
        elif end not in graph._nodes:
            raise SyntaxError("The end node is not in the graph!")
        for node in graph._nodes:
            if node != start:
                score[node] = float('inf') # initialize all scores to inf
            else:
                score[node] = 0 # start node is initalized to 0
        current = start # set currently processed node to start node
        while current != end:
            if current in graph._edges:
                for neighbor in graph._edges[current]:
                    new_score = score[current] + graph._weights[(current, neighbor)]
                    if score[neighbor] > new_score:
                        score[neighbor] = new_score
                        predecessor[neighbor] = current
            unmarked_nodes.remove(current) # mark current node
            min_node = None # find unmarked node with lowest score
            score[min_node] = float('inf')
            for unmarked in unmarked_nodes:
                if score[unmarked] <= score[min_node]: # need equal sign to account to ensure dummy "None" value is replaced
                    min_node = unmarked
            current = min_node # set current to unmarked node with min score
        shortest_path = [end]
        if score[end] != float('inf'):
            start_of_suffix = end
            while predecessor[start_of_suffix] != start:
                shortest_path.append(predecessor[start_of_suffix])
                start_of_suffix = predecessor[start_of_suffix]
            # add start node then reverse list
            shortest_path.append(start)
            shortest_path.reverse()
        else:
            shortest_path = []
    return score[end], shortest_path

def is_disjoint(interval_A, interval_B):
    '''
    this subroutine checks if two intervals intersect with each other; it returns True if
    they do and False otherwise
    input : interval_A - first interval
            interval_B - second interval
    output: is_intersecting - True if interval_A intersects interval_B, False otherwise
    '''
    disjoint = (interval_A[0] > interval_B[1]) or (interval_B[0] > interval_A[1])
    return disjoint

def plot_tubes(ax, cars_to_keep):
    global_vars.curr_tubes = [ax.plot([], [],'b')[0] for _ in range(len(cars_to_keep))]
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
                    global_vars.curr_tubes[i].set_data(xs,ys)


def show_prim_ids(ax, cars_to_keep):
    global_vars.prim_ids_to_show = [ax.text([], [], '') for _ in range(len(cars_to_keep))]
    for i in range(len(cars_to_keep)):
        _,_,x,y = cars_to_keep[i].state
        prim_str = 'None'
        if cars_to_keep[i].prim_queue.len() > 0:
            prim_str = str((cars_to_keep[i].prim_queue.top()[0], int(params.num_subprims*cars_to_keep[i].prim_queue.top()[1])))
        global_vars.prim_ids_to_show[i] = ax.text(x,y, prim_str, color='w', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='red', alpha=0.5), fontsize=10)

def plot_traffic_light_walls(ax, traffic_lights):
    global_vars.walls = [ax.plot([], [],'r')[0] for _ in range(4)]
    if traffic_lights.get_states('horizontal', 'color') == 'red':
        xs = intersection.traffic_light_walls['west']['x']
        ys = intersection.traffic_light_walls['west']['y']
        xs.append(xs[0])
        ys.append(ys[0])
        global_vars.walls[0].set_data(xs,ys)
        xs = intersection.traffic_light_walls['east']['x']
        ys = intersection.traffic_light_walls['east']['y']
        xs.append(xs[0])
        ys.append(ys[0])
        global_vars.walls[1].set_data(xs,ys)
    elif traffic_lights.get_states('vertical', 'color') == 'red':
        xs = intersection.traffic_light_walls['north']['x']
        ys = intersection.traffic_light_walls['north']['y']
        xs.append(xs[0])
        ys.append(ys[0])
        global_vars.walls[2].set_data(xs,ys)
        xs = intersection.traffic_light_walls['south']['x']
        ys = intersection.traffic_light_walls['south']['y']
        xs.append(xs[0])
        ys.append(ys[0])
        global_vars.walls[3].set_data(xs,ys)

def show_license_plates(ax, cars_to_keep):
    global_vars.ids = [ax.text([], [], '') for _ in range(len(cars_to_keep))]
    for i in range(len(cars_to_keep)):
        _,_,x,y = cars_to_keep[i].state
        plate_number = cars_to_keep[i].plate_number
        global_vars.ids[i] = ax.text(x,y,str(plate_number), color='w', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='red', alpha=0.5), fontsize=10)

def plot_boxes(ax, cars_to_keep):
    global_vars.boxes = [ax.plot([], [], 'c')[0] for _ in range(len(cars_to_keep))]
    for i in range(len(cars_to_keep)):
        curr_car = cars_to_keep[i]
        vertex_set,_,_,_ = get_bounding_box(curr_car)
        xs = [vertex[0] for vertex in vertex_set]
        ys = [vertex[1] for vertex in vertex_set]
        xs.append(vertex_set[0][0])
        ys.append(vertex_set[0][1])
        if with_probability(1):
           global_vars.boxes[i].set_data(xs,ys)
        for j in range(i + 1, len(cars_to_keep)):
           if not collision_free(curr_car, cars_to_keep[j]):
                global_vars.boxes[j].set_color('r')
                global_vars.boxes[i].set_color('r')

def show_wavefronts(ax, dt):
    global_vars.honk_waves = [ax.scatter(None,None)]
    honk_xs = []
    honk_ys = []
    radii = []
    intensities = []
    for wave in global_vars.all_wavefronts:
        wave.next(dt)
        honk_x, honk_y, radius, intensity = wave.get_data()
        if intensity > 0:
            honk_xs.append(honk_x)
            honk_ys.append(honk_y)
            radii.append(radius)
            intensities.append(intensity)
        else:
            global_vars.all_wavefronts.remove(wave)

    rgba_colors = np.zeros((len(intensities),4))
    # red color
    rgba_colors[:,0] = 1.0
    # intensities
    rgba_colors[:, 3] = intensities
    global_vars.honk_waves = [ax.scatter(honk_xs, honk_ys, s = radii, lw=1, facecolors='none', color=rgba_colors)]

def honk_randomly(cars_to_keep, prob_on=0.01, prob_off=1):
    for car in cars_to_keep:
        if with_probability(prob_on) and not car.is_honking:
            car.toggle_honk()
            # offset is 600 before scaling
            wave = wavefront.HonkWavefront([car.state[2] + 600*params.car_scale_factor*np.cos(car.state[1]), car.state[3] + 600*params.car_scale_factor*np.sin(car.state[1]), 0, 0], init_energy=100000)
            global_vars.all_wavefronts.add(wave)
        elif with_probability(prob_off) and car.is_honking:
            car.toggle_honk()

def check_for_collisions(cars_to_keep):
    for person in global_vars.pedestrians_to_keep:
        for car in cars_to_keep:
            no_collision,_ = collision_free(person, car)
            if not no_collision and car.state[0] > 1:
                person.is_dead = True

def update_cars(cars_to_keep, dt):
    # determine which cars to keep
    for plate_number in global_vars.all_cars.keys():
        car = global_vars.all_cars[plate_number]
        if car.prim_queue.len() > 0:
            # update cars with primitives
            global_vars.all_cars[plate_number].prim_next(dt)
            # add cars to keep list
            cars_to_keep.append(global_vars.all_cars[plate_number])
        else:
            global_vars.cars_to_remove.add(plate_number)
    # determine which cars to remove
    for plate_number in global_vars.cars_to_remove.copy():
        del global_vars.all_cars[plate_number]
        global_vars.cars_to_remove.remove(plate_number)

def spawn_pedestrian():
    name = random.choice(names)
    age = random.randint(18,70)
    start_node = random.sample(pedestrian_graph.G._sources, 1)[0]
    end_node = random.sample(pedestrian_graph.G._sinks, 1)[0]
    init_state = start_node + (0,0)
    if age > 50:
        pedestrian_type = '2'
    else:
        pedestrian_type = random.choice(['1','2','3','4','5','6'])
    the_pedestrian = pedestrian.Pedestrian(init_state=init_state, pedestrian_type=pedestrian_type, name=name, age=age)
    return name, start_node, end_node, the_pedestrian

def draw_crossings(ax, plt, vertical_lane_color, horizontal_lane_color):
    global_vars.crossing_highlights = []
    lane_colors = {'west': vertical_lane_color, 'east': vertical_lane_color, 'north': horizontal_lane_color, 'south' : horizontal_lane_color}
    for curr_direction in lane_colors:
        x_min = min(intersection.crossing_walls[curr_direction]['x'])
        x_max = max(intersection.crossing_walls[curr_direction]['x'])
        dx = x_max - x_min
        y_min = min(intersection.crossing_walls[curr_direction]['y'])
        y_max = max(intersection.crossing_walls[curr_direction]['y'])
        dy = y_max - y_min
        rect = plt.Rectangle((x_min, y_min), dx, dy, alpha=alt_sin(0.1,0.4,1,global_vars.current_time), color=lane_colors[curr_direction])
        global_vars.crossing_highlights.append(ax.add_artist(rect))


def update_traffic_lights(ax, plt, traffic_lights):
    global_vars.show_traffic_lights = []
    horizontal_light = traffic_lights.get_states('horizontal', 'color')
    vertical_light = traffic_lights.get_states('vertical', 'color')
    for coordinate in traffic_signals.vertical_light_coordinates[vertical_light]:
        x = coordinate[0]
        y = coordinate[1]
        circ = plt.Circle((x,y), radius=6, alpha=alt_sin(0.3,1,100,global_vars.current_time), color=vertical_light[0])
        global_vars.show_traffic_lights.append(ax.add_artist(circ))
    for coordinate in traffic_signals.horizontal_light_coordinates[horizontal_light]:
        x = coordinate[0]
        y = coordinate[1]
        circ = plt.Circle((x,y), radius=6, alpha=alt_sin(0.3,1,100,global_vars.current_time), color=horizontal_light[0])
        global_vars.show_traffic_lights.append(ax.add_artist(circ))

def draw_walk_signs(ax, vertical_fig, horizontal_fig):
    global_vars.walk_signs = []
    for coordinate in traffic_signals.walk_sign_coordinates['vertical']:
        x_corner, y_corner = find_corner_coordinates(0, 0, coordinate[0], coordinate[1], 0, vertical_fig)
        w, h = vertical_fig.size
        global_vars.walk_signs.append(ax.imshow(vertical_fig, extent=(x_corner,x_corner+w, y_corner, y_corner+h)))
    for coordinate in traffic_signals.walk_sign_coordinates['horizontal']:
        x_corner, y_corner = find_corner_coordinates(0, 0, coordinate[0], coordinate[1], 0, horizontal_fig)
        w, h = horizontal_fig.size
        global_vars.walk_signs.append(ax.imshow(horizontal_fig, extent=(x_corner,x_corner+w, y_corner, y_corner+h)))

def within_confines(x,y):
    x_lim, y_lim = intersection.intersection.size
    return x <= x_lim and x >= 0 and y >= 0 and y <= y_lim
