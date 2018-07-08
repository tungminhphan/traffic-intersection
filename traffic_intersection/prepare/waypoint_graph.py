# Waypoint Graphs for Primitive Computation
# Tung M. Phan
# California Institute of Technology
# May 9th, 2018

import numpy as np
if __name__ == '__main__':
    from graph import WeightedDirectedGraph
else:
    from traffic_intersection.prepare.graph import WeightedDirectedGraph
# turn on/off axes
G =  WeightedDirectedGraph()
# a node is of the form (x, y)
edges  = [
             # first lane
             (( 0, 325), (140, 380)),
             (( 140, 380), ( 255, 380)),
             (( 255, 380), ( 400, 380)),
             (( 400, 380), ( 570, 610)),
             (( 570, 610), ( 570, 740)),
             (( 400, 380), ( 635, 610)),
             (( 635, 610), ( 635, 740)),
             # second lane
             (( 0, 325), ( 140, 310)),
             (( 140, 310), ( 255, 310)),
             (( 255, 310), ( 900, 310)),
             (( 900, 310), ( 1040, 340)),
             # third lane
             (( 0, 245), ( 140, 310)),
             (( 0, 245), ( 140, 245)),
             (( 140, 245), ( 255, 245)),
             (( 255, 245), ( 900, 245)),
             (( 900, 245), ( 1040, 255)),
             (( 140, 245), ( 255, 310)),
             (( 140, 310), ( 255, 245)),
             (( 0, 325), ( 140, 245)),
             (( 255, 245), ( 390, 230)),
             (( 255, 245), ( 450, 230)),
             (( 390, 230), ( 430, 30)),
             (( 450, 230), ( 500, 30))
             ]

# add weights, in this case simply the distance
scaling_factor = 0.1
for edge in edges:
    dist = np.sqrt((edge[0][0]-edge[1][0])**2 + (edge[0][1]-edge[1][1])**2) * scaling_factor
    edge = [edge + (dist,)]
    G.add_edges(edge)

G._sources = set([(0, 325), (0, 245)])
G._sinks = set([(430, 30), (500, 30), (1040, 255), (1040, 340), (570, 740), (635, 740)])

def plot_edges(plt, graph, plt_src_snk = False):
    for start_node in G._edges:
        start_x = start_node[0]
        start_y = start_node[1]
        for end_node in G._edges[start_node]:
            end_x = end_node[0]
            end_y = end_node[1]
            dx = end_x - start_x
            dy = end_y - start_y
            # plot transition
            plt.axes().arrow(start_x,start_y, dx, dy, linestyle='dashed', color = 'w',
                    width=edge_width,head_width=head_width, alpha=0.5)
    if plt_src_snk == True:
        node_x = np.zeros(len(graph._sources))
        node_y = np.zeros(len(graph._sources))
        k = 0
        for node in graph._sources:
            node_x[k] = node[0]
            node_y[k] = node[1]
            k += 1
        plt.axes().plot(node_x, node_y, 'ro', markersize=10)
        node_x = np.zeros(len(graph._sinks))
        node_y = np.zeros(len(graph._sinks))
        k = 0
        for node in graph._sinks:
            node_x[k] = node[0]
            node_y[k] = node[1]
            k += 1
        plt.axes().plot(node_x, node_y, 'bo', markersize=10)
        plt.axes().legend(['sources', 'sinks'])

