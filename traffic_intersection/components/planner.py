# Path Planner
# Tung M. Phan
# May 15th, 2018
# California Institute of Technology

import os, sys
sys.path.append("..")
import time, random
import prepare.queue as queue
import prepare.car_waypoint_graph as waypoint_graph
if __name__ == '__main__':
    visualize = True
else:
    visualize = False

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
        if start not in graph._nodes or end not in graph._nodes:
            raise SyntaxError("either the start or end node is not in the graph!")
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

def get_scheduled_times(path, start_time, primitive_graph):
    '''
    this function takes in a path and computes the scheduled times of arrival at the nodes on this path
    input: path - the path whose nodes the user would like to compute the scheduled times of arrival for
    output: a tuple of scheduled times (of arrival) at each node
    '''
    now = get_now(start_time)
    scheduled_times = [now]
    for prev, curr in zip(path[0::1], path[1::1]):
        scheduled_times.append(scheduled_times[-1] + primitive_graph._weights[(prev, curr)])
    return scheduled_times

def time_stamp(path):
    '''
    given a weighted path, this function updates the (node) time_stamps set according to the given path.
    input:   path - weighted path
    output:  modifies time_stamps
    '''
    global time_stamps, start_time, primitive_graph
    scheduled_times = get_scheduled_times(path)
    for k in range(0,len(path)):
        left = max(0, k-1)
        right = min(len(path)-1, k+1)
        stamp = (scheduled_times[left], scheduled_times[right]) # interval stamp
        try: time_stamps[path[k]].add(stamp)
        except KeyError:
            time_stamps[path[k]] = {stamp}

def time_stamp_edge(path, edge_time_stamps, start_time, primitive_graph):
    '''
    given a weighted path, this function updates the edge_time_stamps set according to the given path.
    input:   path - weighted path
    output:  modifies edge_time_stamps
    '''
    scheduled_times = get_scheduled_times(path=path, start_time=start_time, primitive_graph=primitive_graph)
    for k in range(0,len(path)-1):
        left = k
        right = k+1
        edge = (path[left], path[right])
        stamp = (scheduled_times[left], scheduled_times[right]) # interval stamp
        try: edge_time_stamps[edge].add(stamp)
        except KeyError:
            edge_time_stamps[edge] = {stamp}
    return edge_time_stamps

def get_now(start_time):
    '''
    get the current time
    input: None
    output: the current time
    '''
    now = time.time() - start_time
    return now

def is_overlapping(interval_A, interval_B):
    '''
    this subroutine checks if two intervals intersect with each other; it returns True if
    they do and False otherwise
    input : interval_A - first interval
            interval_B - second interval
    output: is_intersecting - True if interval_A intersects interval_B, False otherwise
    '''
    is_intersecting = interval_A[1] >= interval_B[0] and interval_A[0] <= interval_B[1] or interval_B[1] >= interval_A[0] and interval_B[0] <= interval_A[1]
    return is_intersecting

def nodes_are_safe(path):
    now = get_now(start_time)
    scheduled_times = [now]
    k = 0
    for curr, nxt in zip(path[0::1], path[1::1]):
        scheduled_times.append(scheduled_times[-1] + primitive_graph._weights[(curr, nxt)])
        left = max(0, k-1)
        right = -1
        curr_interval = (scheduled_times[left], scheduled_times[right]) # next interval to check
        if curr in time_stamps: # if current loc is already stamped
            for interval in time_stamps[curr]:
                if is_overlapping(curr_interval, interval):
                    # if the two intervals overlap
                    return curr # return node with conflict
        k += 1
    # now check last node
    left = max(0, k-1)
    right = -1
    curr_interval = (scheduled_times[left], scheduled_times[right]) # last interval
    curr = path[-1]
    if curr in time_stamps: # if current loc is already stamped
        for interval in time_stamps[curr]:
            if is_overlapping(curr_interval, interval):
                # if the two intervals overlap
                return curr # return node with conflict
    return True

def is_safe(path):
    now = get_now(start_time)
    scheduled_times = [now]
    current_edge_idx = 0 
    for left_node, right_node in zip(path[0::1], path[1::1]):
        edge = (left_node, right_node)
        scheduled_times.append(scheduled_times[-1] + primitive_graph._weights[edge])
        left_time = scheduled_times[-2]
        right_time = scheduled_times[-1]
        curr_interval = (left_time, right_time) # next interval to check
        if edge in edge_time_stamps: # if current loc is already stamped
            for interval in edge_time_stamps[edge]:
                if is_overlapping(curr_interval, interval):
                    # if the two intervals overlap
                    return current_edge_idx # return node with conflict
        current_edge_idx += 1
    return True

def print_state():
    print('The current request queue state is')
    request_queue.print_queue()

def process_request():
    '''
    TODO: write comments
    '''
    global request_queue
    if request_queue.len() == 0: # if request queue is empty, do nothing
        pass
    else: # else start processing next item in queue
        request = request_queue.pop() # take the next request in line
        path_score, shortest_path = dijkstra(request['start'], request['end'], primitive_graph) # find the shortest path for it
        if path_score == float('inf'): # if there is no path between start and end
            print('the request to go from', request['start'], 'to', request['end'], 'was rejected due to the start and end nodes being unreachable')
        else:
            safety_check = is_safe(shortest_path) # check if shortest_path is safe
            if safety_check == True: # if it is, honor request
                time_stamp_edge(path = shortest_path, time_stamps = time_stamps, start_time = time_stamp, primitive_graph = primitive_graph) # timestamp request
                print('the request to go from', request['start'],  'to', request['end'], '(with license plate ' + str(request['license_plate']) + ') was fully processed at time',round(time.time()-start_time,2))
            else:
                    service_path = shortest_path[0:safety_check]
                    time_stamp_edge(service_path)
                    print('the request to go from', request['start'], 'to', request['end'],'along the path', shortest_path, 'was only granted up to ' + str(shortest_path[safety_check]))
                    request['start']= shortest_path[safety_check]
                    request_queue.enqueue(request)

def generate_license_plate():
    import string
    choices = string.digits + string.ascii_uppercase
    plate_number = ''
    for i in range(0,7):
        plate_number = plate_number + random.choice(choices)
    return plate_number

def planner_update():
    process_request()
    if random.random() >= arrival_chance:
        start = random.choice(list(primitive_graph._sources))
        end = random.choice(list(primitive_graph._sinks))
        car_id = generate_license_plate()
        request = {'start': start, 'end': end, 'license_plate': car_id}
        print('the car with license plate number', request['license_plate'], 'requests to go from', request['start'], 'to', request['end'])
        request_queue.enqueue(request)
    print_state()
    time.sleep(random.random())

                   ######################################################
                   ######################################################
                   ###                                                ###
                   ###                    SIMULATION                  ###
                   ###                                                ###
                   ######################################################
                   ######################################################


if visualize:
    # define primitive_graph
    primitive_graph = waypoint_graph.G
    # time_stamps a dictionary with nodes as keys containing reserved intervals
    time_stamps = {}
    # edge_time_stamps a dictionary with edges as keys containing reserved intervals
    edge_time_stamps = {}
    # start_time is the start time of the simulation
    start_time = time.time()
    # request queue
    request_queue = queue.Queue()
    # probability of new arrival
    arrival_chance = 0.7

    while True:
        planner_update()
