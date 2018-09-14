# Path Planner
# Tung M. Phan
# May 15th, 2018
# California Institute of Technology

import sys
sys.path.append('..')
import random
import components.traffic_signals as traffic_signals
import prepare.queue as queue
import prepare.car_waypoint_graph as waypoint_graph
import primitives.tubes
from primitives.load_primitives import get_prim_data
import numpy as np
import assumes.params as params
import variables.global_vars as global_vars

collision_dictionary = np.load('prepare/collision_dictionary.npy').item()
edge_to_prim_id = np.load('prepare/edge_to_prim_id.npy').item()

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

def make_transit(node):
    return (0, node[1], node[2], node[3])

def find_transit(path, graph, effective_time, plate_number, traffic_lights):
    depart = path[0]
    arrive = path[-1]
    potential_transits = path[1:-1][::-1] # take all nodes between depart and arrive and do a reversal
    head_ok = False
    tail_ok = False
    if len(potential_transits) > 0:
        for potential_transit in potential_transits:
            transit = make_transit(potential_transit)
            if transit in graph._nodes:
                # check if tail is a path
                _, tail = dijkstra(transit, arrive, graph)
                tail_ok = len(tail) > 0
                # check if head is a path
                _, head = dijkstra(depart, transit, graph)
                # check if head is safe
                if len(head) > 0:
                    head_ok = head_is_safe(path=head, effective_time=effective_time, graph=graph, plate_number=plate_number, traffic_lights=traffic_lights)
                if tail_ok and head_ok:
                    return head
    return None

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

def extract_destinations(request):
    depart, arrive = request[1], request[2]
    return depart, arrive

def extract_car_info(request):
    plate_number, the_car = request[0], request[3]
    return plate_number, the_car

def subprim_is_safe(subprim_id, subprim_interval, plate_number, traffic_lights):
    for box in collision_dictionary[subprim_id]:
        if isinstance(box, str):
            if not crossing_safe(subprim_interval, box, traffic_lights):
                return False
        elif box in global_vars.time_table: # if current conflicting prim is already scheduled
            for car_id in global_vars.time_table[box]:
                if car_id != plate_number:
                    interval = global_vars.time_table[box][car_id]
                    if not is_disjoint(subprim_interval, interval): # if the two intervals overlap
                        return False
    return True

def shift_interval(interval, dt):
    return (interval[0] + dt, interval[1] + dt)

def crossing_safe(interval, which_light, traffic_lights):
    interval_length = interval[1] - interval[0]
    first_time = interval[0]
    is_safe = False
    if which_light in {'north', 'south', 'west', 'east'}:
        predicted_state = traffic_lights.predict(first_time, True) # if horizontal
        if which_light == 'north' or which_light == 'south': # if vertical
            predicted_state = traffic_lights.get_counterpart(predicted_state) # vertical light
        color, time = predicted_state
        remaining_time = traffic_lights._max_time[color] - time
        is_safe = (color == 'yellow' and remaining_time > interval_length) or (color == 'green' and remaining_time + traffic_lights._max_time['yellow'] > interval_length)
    else: # if invisible wall is due to crossing
        predicted_state = traffic_lights.predict(first_time, True) # if horizontal
        if which_light == 'crossing_west' or which_light == 'crossing_east': # if vertical
            predicted_state = traffic_lights.get_counterpart(predicted_state) # vertical light
        color, time = predicted_state
        is_safe = not (color == 'green' and time <= traffic_lights._max_time['green']/3)
        #is_safe = not (color == 'green' and time <= traffic_lights._max_time['green']/3 or color == 'yellow' and time >= traffic_lights._max_time['yellow']/5)
    return is_safe

def get_scheduled_intervals(path, effective_time, graph, complete_path=True):
    now = effective_time
    path_prims = path_to_primitives(path)
    scheduled_intervals = dict()
    time_shift = effective_time
    for prim_id in path_prims:
        prim_time = get_prim_data(prim_id, 't_end')[0]
        dt = prim_time / params.num_subprims
        for subprim_id in range(params.num_subprims):
            subprim = (prim_id, subprim_id)
            subprim_interval = shift_interval((0, dt), time_shift)
            scheduled_intervals[subprim] = subprim_interval
            time_shift += dt
    if not complete_path:
        last_subprim = (path_prims[-1], params.num_subprims-1)
        last_interval = scheduled_intervals[last_subprim]
        scheduled_intervals[last_subprim] = (last_interval[0], float('inf'))
    return scheduled_intervals

def complete_path_is_safe(path, effective_time, graph, plate_number, traffic_lights):
    scheduled_intervals = get_scheduled_intervals(path, effective_time, graph)
    for subprim in scheduled_intervals:
        if not subprim_is_safe(subprim_id=subprim, subprim_interval=scheduled_intervals[subprim], plate_number=plate_number,  traffic_lights=traffic_lights):
            return False
    return True

def head_is_safe(path, effective_time, graph, plate_number, traffic_lights):
    scheduled_intervals = get_scheduled_intervals(path=path, effective_time=effective_time, graph=graph, complete_path=False)
    for subprim in scheduled_intervals:
        if not subprim_is_safe(subprim_id=subprim, subprim_interval=scheduled_intervals[subprim], plate_number=plate_number,traffic_lights=traffic_lights):
            return False
    return True

def path_to_primitives(path):
    primitives = []
    for node_s, node_e  in zip(path[:-1], path[1:]):
            next_prim_id = edge_to_prim_id[(node_s, node_e)]
            primitives.append(next_prim_id)
    return primitives

def refresh_effective_times(plate_number, effective_times):
    if plate_number in effective_times.keys():
        effective_times[plate_number] = max(global_vars.current_time, effective_times[plate_number])
    else:
        effective_times[plate_number] = global_vars.current_time

def send_prims_and_update_effective_times(plate_number, the_car, cars, effective_times, path_prims):
    if plate_number not in cars:
        cars[plate_number] = the_car # add the car
    for prim_id in path_prims:
        cars[plate_number].prim_queue.enqueue((prim_id, 0))
        effective_times[plate_number] += get_prim_data(prim_id, 't_end')[0]

def update_table(scheduled_intervals, plate_number):
    for sub_prim in scheduled_intervals:
        interval = scheduled_intervals[sub_prim]
        if not sub_prim in global_vars.time_table:
            global_vars.time_table[sub_prim] = {plate_number: interval}
        else:
            global_vars.time_table[sub_prim][plate_number] = interval

def make_wait(the_car, plate_number, waiting, wait_subprim):
    the_car.prim_queue.enqueue((-1,0))
    waiting[plate_number] = wait_subprim

def unwait(the_car, plate_number, waiting):
    the_car.prim_queue.remove((-1,0))
    if plate_number in waiting:
        wait_subprim = waiting[plate_number]
        del global_vars.time_table[wait_subprim][plate_number]
        del waiting[plate_number]

def serve_request(request, graph, effective_times, cars, waiting, traffic_lights):
    depart, arrive = extract_destinations(request)
    plate_number, the_car = extract_car_info(request)
    refresh_effective_times(plate_number, effective_times)
    effective_time = effective_times[plate_number]
    _, path = dijkstra(depart, arrive, graph)
    if complete_path_is_safe(path=path, effective_time=effective_time, graph=graph, plate_number=plate_number, traffic_lights=traffic_lights):
        unwait(the_car, plate_number, waiting)
        scheduled_intervals = get_scheduled_intervals(path, effective_time, graph)
        update_table(scheduled_intervals=scheduled_intervals,plate_number=plate_number)
        path_prims = path_to_primitives(path)
        send_prims_and_update_effective_times(plate_number, the_car, cars, effective_times, path_prims)
    else:
        head = find_transit(path=path, graph=graph,  effective_time=effective_time,  plate_number=plate_number,  traffic_lights=traffic_lights)
        if head != None:
            unwait(the_car, plate_number, waiting)
            transit = head[-1]
            scheduled_intervals = get_scheduled_intervals(path=head, effective_time=effective_time, graph=graph, complete_path=False)
            update_table(scheduled_intervals=scheduled_intervals,plate_number=plate_number)
            path_prims = path_to_primitives(head)
            send_prims_and_update_effective_times(plate_number=plate_number,the_car=the_car,cars=cars,effective_times=effective_times,path_prims=path_prims)
            wait_subprim = (path_prims[-1],params.num_subprims-1)
            make_wait(the_car=the_car,plate_number=plate_number,waiting=waiting,wait_subprim=wait_subprim)
            global_vars.request_queue.enqueue((plate_number, transit, arrive, the_car))
        elif plate_number not in cars and depart[0] == 0:
            path_prims = path_to_primitives(path)
            wait_subprim = (path_prims[0], 0)
            wait_interval = (global_vars.current_time, float('inf'))
            if subprim_is_safe(wait_subprim, wait_interval, plate_number, traffic_lights):
                cars[plate_number] = the_car # add the car
                scheduled_intervals = dict()
                scheduled_intervals[wait_subprim] = wait_interval
                update_table(scheduled_intervals=scheduled_intervals,plate_number=plate_number)
                make_wait(the_car=the_car,plate_number=plate_number,waiting=waiting,wait_subprim=wait_subprim)
                global_vars.request_queue.enqueue(request)
        elif plate_number in cars:
            global_vars.request_queue.enqueue(request)
