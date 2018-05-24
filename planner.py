# Path Planner
# Tung M. Phan
# May 16th, 2018
# California Institute of Technology

# looks at current state
# 
# Input: Current State (Traffic Light Signal etc)
#        Graph?
# Output: A set of nodes or primitives to follow
# how do you define the graph structure 
# map is global

import prepare.graph as graph
import time


input_graph = graph.WeightedDirectedGraph()
input_graph.add_edges([('1', '2', 2)])
input_graph.add_edges([('1', '3', 4)])
input_graph.add_edges([('2', '3', 1)])
input_graph.add_edges([('2', '4', 4)])
input_graph.add_edges([('2', '5', 2)])
input_graph.add_edges([('3', '5', 3)])
input_graph.add_edges([('5', '4', 3)])
input_graph.add_edges([('4', '6', 2)])
input_graph.add_edges([('5', '6', 2)])

def traffic_controller():
    global traffic_signal, queues
    # while True
    # check traffic light
    # try to empty car
    pass



world_map = {'A1': {'to': {'A2': {'weight': 1, 'timestamp': 5}, 'A3': {'weight': 2,
    'timestamp': 2}}}}
print('The timestamp is ' + str(world_map['A1']['to']['A2']['timestamp']) + " units")


def to_planning_graph(graph):
    return True

def check_collision(path):
    score = {}

    for node in path:
        score[node] = score[previous_node] + world_map[new_node]
        if abs(world_map[node][timestamp] - score[node]) >= safety:
            pass

    return True

def enqueue():
    return True

def timestamp_path(path):
    for node in path:
        world_map[node][timestamp] = time.clock() + cummulative_weight



def compute_path(start, end, world_map):
    global traffic_signal, itineraries, work_queue, clock
    tentative_cost, tentative_path = dijkstra(start, end, world_map)
    if check_collision():
        destination = check_collision()
        marked_execution(partial_path)
    # should mark an attempt time and a release time


    return destination




    # convert graph to something that can be used by



    # assumes world graph given is correct...
    # guarantees a marked graph output, in the process updating world map
    # as one enters an intersection, all information that is available to him is
    # traffic light, how other cars move...? he should also have an idea of what to
    # do, a purpose in the intersection..
    # since we already have Dijkstra i guess the question would be how the edges shouldbe
    # relabelled also what happens when blocked?? requeued?
    # how many queues are needed? one for each direction?

    # input
    # start node, end node
    # graph # outputs shortest path
    # take the waypoint if it hasn't been taken... otherwise, paint it
    # paint the path that has been c
    # return painted graph
    # find shortest path
    # paint it and return
    # compute next move
    print("check")

def dijkstra(start, end, graph):
    '''
    This code takes in a weighted directed graph, a start node, an end node and outputs
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
time_stamps = {}
def time_stamp(path):
    global time_stamps
    stamp = time.clock()
    first = path[0]
    try: time_stamps[first].add(stamp)
    except KeyError:
        time_stamps[first] = {stamp}
    for prev, curr in zip(path[0::1], shortest_path[1::1]):
        stamp = stamp + input_graph._weights[(prev,curr)]
        try: time_stamps[curr].add(stamp)
        except KeyError:
            time_stamps[curr] = {stamp}

score, shortest_path = dijkstra('1', '6', input_graph)
print('The cost is: ' + str(score))
print('The path is: ' + str(shortest_path))

time_stamp(shortest_path)
print(time_stamps)
time.sleep(2)
score, shortest_path = dijkstra('2', '5', input_graph)
time_stamp(shortest_path)
print(time_stamps)
