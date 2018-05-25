# Path Planner
# Tung M. Phan
# May 16th, 2018
# California Institute of Technology

import time
import prepare.graph as graph
import numpy as np
np.set_printoptions(precision=2)


def traffic_controller():
    global traffic_signal, queues
    # while True
    # check traffic light
    # try to empty car
    pass

def check_collision(path):
    score = {}
    for node in path:
        score[node] = score[previous_node] + world_map[new_node]
        if abs(world_map[node][timestamp] - score[node]) >= safety:
            pass
    return True

def compute_path(start, end, world_map):
    global traffic_signal, itineraries, work_queue, clock
    tentative_cost, tentative_path = dijkstra(start, end, world_map)
    if check_collision():
        destination = check_collision()
        marked_execution(partial_path)
    # should mark an attempt time and a release time
    return destination

def dijkstra(start, end, graph):
    '''
    This function takes in a weighted directed graph, a start node, an end node and outputs
    the shortest path from the start node to the end node on that graph
    Input:  start - start node
            end - end node
            graph - weighted directed graph
    Output: shortest path from start to end node

    '''
    score = {}
    predecessor = {}
    unmarked_nodes = graph._nodes.copy() # create a copy of set of nodes in graph
    if start not in graph._nodes or end not in graph._nodes:
        raise SyntaxError("Either the start or end node is not in the graph!")

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
        # find unmarked node with lowest score
        min_node = None
        score[min_node] = float('inf')
        for unmarked in unmarked_nodes:
            if score[unmarked] < score[min_node]:
                min_node = unmarked
        current = min_node # set current to unmarked node with min score
    start_of_prefix = end
    shortest_path = [end]
    while predecessor[start_of_prefix] != start:
        shortest_path.append(predecessor[start_of_prefix])
        start_of_prefix = predecessor[start_of_prefix]
    # add start node then reverse list
    shortest_path.append(start)
    shortest_path.reverse()
    return score[end], shortest_path

def get_scheduled_times(path):
    '''
    This function takes in a path and computes the scheduled times of arrival at the nodes on this path
    Input: path - the path whose nodes the user would like to compute the scheduled times of arrival for
    Output: a tuple of scheduled times
    '''
    global start_time
    now = round(time.time() - start_time, 2) # round to 2nd decimal place
    scheduled_times = [now]
    for prev, curr in zip(path[0::1], shortest_path[1::1]):
        scheduled_times.append(scheduled_times[-1] + primitive_graph._weights[(prev, curr)])
    return scheduled_times


def time_stamp(path):
    '''
    Given a weighted path, this function updates the time_stamps set according to the given path.

    Input:   path - weighted path
    Output:  modifies time_stamps
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

def safety_check(path):
    global start_time
    now = round(time.time() - start_time, 2) # round to 2nd decimal place
    scheduled_times = [now]
    for prev, curr in zip(path[0::1], shortest_path[1::1]):
        scheduled_times.append(scheduled_times[-1] + primitive_graph._weights[(prev, curr)])
        if curr in time_stamps:
            current_time = scheduled_times[-1]
            for interval in time_stamps[curr]:
                if current_time >= interval[0] and current_time <= interval[1]:
                    print(curr)
                    return False
    return True

    ######################################################
    ######################################################
    ##                                                  ##
    ##                   SIMULATION                     ##
    ##                                                  ##
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

# start_time is the start time of the simulation
start_time = time.time()

score, shortest_path = dijkstra('1', '6', primitive_graph)


time_stamp(shortest_path)
print(time_stamps)
time.sleep(0.1)
score, shortest_path = dijkstra('2', '6', primitive_graph)
time_stamp(shortest_path)
print(time_stamps)
print(safety_check(shortest_path))

