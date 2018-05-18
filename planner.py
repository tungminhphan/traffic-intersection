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
input_graph.add_edges([('a', 'b', 10), ('a', 'c', 2), ('c', 'z', 3)])
input_graph.add_edges([('b', 'z', 5), ('b', 'c', 1), ('a', 'z', 19)])

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
    predecessor = {}
    unmarked_nodes = graph._nodes
    if start not in graph._nodes or end not in graph._nodes:
        raise SyntaxError("Either the start or end node is not in the graph!")

    for node in graph._nodes:
        if node != start:
            score[node] = float('inf') # initialize all scores to inf
        else:
            score[node] = 0 # start node is initalized to 0
    current = start # set currently processed node to start node

    while current != end:
        for neighbor in graph._edges[current]:
            if score[neighbor] > score[current] + graph._weights[(current, neighbor)]:
                score[neighbor] = score[current] + graph._weights[(current, neighbor)]
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

input_graph.print_graph()
score, shortest_path = dijkstra('a', 'z', input_graph)
print('The cost is: ' + str(score))
print('The path is: ' + str(shortest_path))
