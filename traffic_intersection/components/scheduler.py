# Scheduler Class
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
from prepare.helper import *
collision_dictionary = np.load('../prepare/collision_dictionary.npy').item()
edge_to_prim_id = np.load('../prepare/edge_to_prim_id.npy').item()

class Scheduler:
    '''Kinematic car class

    init_state is [vee, theta, x, y], where vee, theta, x, y are the velocity, orientation, and
    coordinates of the car respectively
    '''
    def __init__(self):
        self._time_table = dict()
        self._request_queue = queue.Queue()
        self._waiting_dict = dict()
        self._effective_times = dict()

    def make_transit(self, node):
        return (0, node[1], node[2], node[3])

    def find_transit(self, path, graph, effective_time, the_car, traffic_lights):
        depart = path[0]
        arrive = path[-1]
        potential_transits = path[1:-1][::-1] # take all nodes between depart and arrive and do a reversal
        head_ok = False
        tail_ok = False
        if len(potential_transits) > 0:
            for potential_transit in potential_transits:
                transit = self.make_transit(potential_transit)
                if transit in graph._nodes:
                    # check if tail is a path
                    _, tail = dijkstra(transit, arrive, graph)
                    tail_ok = len(tail) > 0
                    # check if head is a path
                    _, head = dijkstra(depart, transit, graph)
                    # check if head is safe
                    if len(head) > 0:
                        head_ok = self.head_is_safe(path=head, effective_time=effective_time, graph=graph, the_car = the_car, traffic_lights=traffic_lights)
                    if tail_ok and head_ok:
                        return head
        return None

    def extract_destinations(self, request):
        depart, arrive = request[0], request[1]
        return depart, arrive

    def extract_car_info(self, request):
        the_car = request[2]
        return the_car

    def subprim_is_safe(self, subprim_id, subprim_interval, the_car, traffic_lights):
        for box in collision_dictionary[subprim_id]:
            if isinstance(box, str):
                if not self.crossing_safe(subprim_interval, box, traffic_lights):
                    return False
            elif box in self._time_table: # if current conflicting prim is already scheduled
                for car_id in self._time_table[box]:
                    if car_id != the_car.plate_number:
                        interval = self._time_table[box][car_id]
                        if not is_disjoint(subprim_interval, interval): # if the two intervals overlap
                            return False
        return True

    def shift_interval(self, interval, dt):
        return (interval[0] + dt, interval[1] + dt)

    def crossing_safe(self, interval, which_light, traffic_lights):
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

    def get_scheduled_intervals(self, path, effective_time, graph, complete_path=True):
        path_prims = self.path_to_primitives(path)
        scheduled_intervals = dict()
        time_shift = effective_time
        for prim_id in path_prims:
            prim_time = get_prim_data(prim_id, 't_end')[0]
            dt = prim_time / params.num_subprims
            for subprim_id in range(params.num_subprims):
                subprim = (prim_id, subprim_id)
                subprim_interval = self.shift_interval((0, dt), time_shift)
                scheduled_intervals[subprim] = subprim_interval
                time_shift += dt
        if not complete_path:
            last_subprim = (path_prims[-1], params.num_subprims-1)
            last_interval = scheduled_intervals[last_subprim]
            scheduled_intervals[last_subprim] = (last_interval[0], float('inf'))
        return path_prims, scheduled_intervals

    def complete_path_is_safe(self, path, effective_time, graph, the_car, traffic_lights):
        _,scheduled_intervals = self.get_scheduled_intervals(path, effective_time, graph)
        for subprim in scheduled_intervals:
            if not self.subprim_is_safe(subprim_id=subprim, subprim_interval=scheduled_intervals[subprim], the_car=the_car, traffic_lights=traffic_lights):
                return False
        return True

    def head_is_safe(self, path, effective_time, graph, the_car, traffic_lights):
        _,scheduled_intervals = self.get_scheduled_intervals(path=path, effective_time=effective_time, graph=graph, complete_path=False)
        for subprim in scheduled_intervals:
            if not self.subprim_is_safe(subprim_id=subprim, subprim_interval=scheduled_intervals[subprim], the_car=the_car,traffic_lights=traffic_lights):
                return False
        return True

    def path_to_primitives(self, path):
        primitives = []
        for node_s, node_e  in zip(path[:-1], path[1:]):
                next_prim_id = edge_to_prim_id[(node_s, node_e)]
                primitives.append(next_prim_id)
        return primitives

    def refresh_effective_times(self, the_car):
        if the_car.plate_number in self._effective_times.keys():
            self._effective_times[the_car.plate_number] = max(global_vars.current_time, self._effective_times[the_car.plate_number])
        else:
            self._effective_times[the_car.plate_number] = global_vars.current_time

    def send_prims_and_update_effective_times(self, the_car, path_prims):
        if the_car.plate_number not in global_vars.all_cars:
            global_vars.all_cars[the_car.plate_number] = the_car # add the car
        for prim_id in path_prims:
            global_vars.all_cars[the_car.plate_number].prim_queue.enqueue((prim_id, 0))
            self._effective_times[the_car.plate_number] += get_prim_data(prim_id, 't_end')[0]

    def update_table(self, scheduled_intervals, the_car):
        for sub_prim in scheduled_intervals:
            interval = scheduled_intervals[sub_prim]
            if not sub_prim in self._time_table:
                self._time_table[sub_prim] = {the_car.plate_number: interval}
            else:
                self._time_table[sub_prim][the_car.plate_number] = interval

    def make_wait(self, the_car, wait_subprim):
        the_car.prim_queue.enqueue((-1,0))
        self._waiting_dict[the_car.plate_number] = wait_subprim

    def unwait(self, the_car):
        the_car.prim_queue.remove((-1,0))
        if the_car.plate_number in self._waiting_dict:
            wait_subprim = self._waiting_dict[the_car.plate_number]
            del self._time_table[wait_subprim][the_car.plate_number]
            del self._waiting_dict[the_car.plate_number]

    def clear_stamps(self):
        for sub_prim in self._time_table.copy():
            for plate_number in self._time_table[sub_prim].copy():
                interval = self._time_table[sub_prim][plate_number]
                if interval[1] < global_vars.current_time:
                    del self._time_table[sub_prim][plate_number]
            if len(self._time_table[sub_prim]) == 0:
                del self._time_table[sub_prim]

    def serve(self, graph, traffic_lights):
        request = self._request_queue.pop()
        depart, arrive = self.extract_destinations(request)
        the_car = self.extract_car_info(request)
        self.refresh_effective_times(the_car)
        effective_time = self._effective_times[the_car.plate_number]
        _, path = dijkstra(depart, arrive, graph)
        if self.complete_path_is_safe(path=path, effective_time=effective_time, graph=graph, the_car=the_car, traffic_lights=traffic_lights):
            self.unwait(the_car)
            path_prims, scheduled_intervals = self.get_scheduled_intervals(path, effective_time, graph)
            self.update_table(scheduled_intervals=scheduled_intervals,the_car=the_car)
            self.send_prims_and_update_effective_times(the_car, path_prims)
        else:
            head = self.find_transit(path=path, graph=graph,  effective_time=effective_time,  the_car=the_car,  traffic_lights=traffic_lights)
            if head != None:
                self.unwait(the_car)
                transit = head[-1]
                path_prims, scheduled_intervals = self.get_scheduled_intervals(path=head, effective_time=effective_time, graph=graph, complete_path=False)
                self.update_table(scheduled_intervals=scheduled_intervals,the_car=the_car)
                self.send_prims_and_update_effective_times(the_car=the_car,path_prims=path_prims)
                wait_subprim = (path_prims[-1],params.num_subprims-1)
                self.make_wait(the_car=the_car,wait_subprim=wait_subprim)
                self._request_queue.enqueue((transit, arrive, the_car))
            elif the_car.plate_number not in global_vars.all_cars and depart[0] == 0:
                path_prims = self.path_to_primitives(path)
                wait_subprim = (path_prims[0], 0)
                wait_interval = (global_vars.current_time, float('inf'))
                if self.subprim_is_safe(wait_subprim, wait_interval, the_car, traffic_lights):
                    global_vars.all_cars[the_car.plate_number] = the_car # add the car
                    scheduled_intervals = dict()
                    scheduled_intervals[wait_subprim] = wait_interval
                    self.update_table(scheduled_intervals=scheduled_intervals,the_car=the_car)
                    self.make_wait(the_car=the_car,wait_subprim=wait_subprim)
                    self._request_queue.enqueue(request)
            elif the_car.plate_number in global_vars.all_cars:
                self._request_queue.enqueue(request)
