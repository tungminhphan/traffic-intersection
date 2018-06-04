# Path Planner
# Tung M. Phan
# May 16th, 2018
# California Institute of Technology

import time, random
import prepare.graph as graph
import prepare.queue as queue
import pdb

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

def get_scheduled_times(path):
    '''
    this function takes in a path and computes the scheduled times of arrival at the nodes on this path
    input: path - the path whose nodes the user would like to compute the scheduled times of arrival for
    output: a tuple of scheduled times (of arrival) at each node
    '''
    now = get_now()
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

def time_stamp_edge(path):
    '''
    given a weighted path, this function updates the edge_time_stamps set according to the given path.
    input:   path - weighted path
    output:  modifies edge_time_stamps
    '''
    global time_stamps, start_time, primitive_graph
    scheduled_times = get_scheduled_times(path)
    for k in range(0,len(path)-1):
        left = k
        right = k+1
        edge = (path[left], path[right])
        stamp = (scheduled_times[left], scheduled_times[right]) # interval stamp
        try: edge_time_stamps[edge].add(stamp)
        except KeyError:
            edge_time_stamps[edge] = {stamp}

def get_now():
    '''
    get the current time
    input: None
    output: the current time
    '''
    global start_time
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
    now = get_now()
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
    now = get_now()
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
    request_queue.print()
    #for edge in edge_time_stamps:
    #    print(edge[0], '->', edge[1])
    #    for interval in edge_time_stamps[edge]:
    #        print('(', round(interval[0], 2),',',round(interval[1],2),')')
                   ######################################################
                   ######################################################
                   ###                                                ###
                   ###                    SIMULATION                  ###
                   ###                                                ###
                   ######################################################
                   ######################################################
# define primitive_graph
primitive_graph = graph.WeightedDirectedGraph()
primitive_graph.add_edges([('1', '2', 2.5)])
primitive_graph.add_edges([('1', '3', 4.2)])
primitive_graph.add_edges([('2', '3', 3.5)])
primitive_graph.add_edges([('2', '4', 4.6)])
primitive_graph.add_edges([('2', '5', 2.1)])
primitive_graph.add_edges([('3', '5', 3.8)])
primitive_graph.add_edges([('5', '4', 3.9)])
primitive_graph.add_edges([('4', '6', 2.1)])
primitive_graph.add_edges([('5', '6', 2.5)])

# time_stamps a dictionary with nodes as keys containing reserved intervals
time_stamps = {}

# edge_time_stamps a dictionary with edges as keys containing reserved intervals
edge_time_stamps = {}

# start_time is the start time of the simulation
start_time = time.time()

# request queue
request_queue = queue.Queue()

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
                time_stamp_edge(shortest_path) # timestamp request
                print('the request to go from', request['start'],  'to', request['end'], '(car_id ' + str(request['car_id']) + ') was fully processed at time',round(time.time()-start_time,2))
            else:
                    service_path = shortest_path[0:safety_check]
                    time_stamp_edge(service_path)
                    print('the request to go from', request['start'], 'to', request['end'],'along the path', shortest_path, 'was only granted up to ' + str(shortest_path[safety_check]))
                    request['start']= shortest_path[safety_check]
                    request_queue.enqueue(request)
while True:
    process_request()
    if random.random() >= 0.9 and request_queue.len() < 5:
        start = random.choice(list(primitive_graph._nodes))
        end = random.choice(list(primitive_graph._nodes))
        import string
        car_id = random.choice(string.ascii_letters)
        request = {'start': start, 'end': end, 'car_id': car_id}
        print('a new request', request, 'has been added')
        request_queue.enqueue(request)
    print_state()
    time.sleep(random.random())
