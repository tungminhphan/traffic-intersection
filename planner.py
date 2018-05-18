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
input_graph = graph.WeightedDirectedGraph()
input_graph.add_edges([('a', 'b', 10), ('a', 'c', 2), ('z', 'c', 3)])

def compute_path(start, end, graph):
    # input
    # start node, end node
    # graph
    # outputs shortest path
    global world_map
    # take the waypoint if it hasn't been taken... otherwise, paint it
    # paint the path that has been c
    # return painted graph
    # find shortest path
    # paint it and return
    # compute next move
    print("check")

def dijkstra(start, end, graph):
    score = {}
    if start not in graph._nodes or end not in graph._nodes:
        raise SyntaxError("Either the start or end node is not in the graph!")
    for node in graph._nodes:
        if node != start:
            score[node] = float('inf') # initialize all scores to 0
        else
            score[node] = 0
    return score

input_graph.print_graph()
score = dijkstra('a', 'x', input_graph)
print(score)
