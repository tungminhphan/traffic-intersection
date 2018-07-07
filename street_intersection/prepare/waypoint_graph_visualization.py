# Waypoint Graphs for Primitive Computation
# Tung M. Phan
# California Institute of Technology
# May 9th, 2018

import os
from numpy import cos, sin, pi
import numpy as np
if __name__ == '__main__':
    from graph import DirectedGraph
else:
    from prepare.graph import DirectedGraph
import matplotlib.pyplot as plt
from PIL import Image
dir_path = os.path.dirname(os.path.realpath(__file__))
intersection_fig = os.path.dirname(dir_path) + "/components/imglib/intersection.png"
fig = plt.figure()

# turn on/off axes
plt.axis("on")
background = Image.open(intersection_fig)
plt.imshow(background, origin="Lower")

markersize = 15
car_width = 8
edge_width = 0.5
head_width = 10
L = 50 # distance between two axles
show_car = True # show car (as arrow) or marker
def plot_car(node):
    global L
    v, theta, x, y = node
    if show_car:
        plt.arrow(x, y, L*cos(theta), L*sin(theta), color='b', width=car_width,alpha=0.5)
    else:
        plt.plot(x,y, 'b.', markersize=markersize, alpha=0.5)

G = DirectedGraph()

# horizontal direction
G.add_edges([
             # first lane
             (( 0, 325), (140, 380)),
             (( 140, 380), ( 280, 380)),
             (( 280, 380), ( 400, 380)),
             (( 400, 380), ( 500, 470)),
             (( 500, 470), ( 570, 610)),
             (( 570, 610), ( 570, 740)),
             (( 400, 380), ( 500, 470)),
             (( 500, 470), ( 635, 610)),
             (( 635, 610), ( 635, 740)),
             # second lane
             (( 0, 325), ( 140, 310)),
             (( 140, 310), ( 280, 310)),
             (( 280, 310), ( 400, 310)),
             (( 400, 310), ( 520, 310)),
             (( 520, 310), ( 640, 310)),
             (( 640, 310), ( 750, 310)),
             (( 750, 310), ( 900, 310)),
             (( 900, 310), ( 1040, 340)),
             (( 900, 310), ( 1040, 340)),
             # third lane
             (( 0, 245), ( 140, 310)),
             (( 0, 245), ( 140, 245)),
             (( 140, 245), ( 280, 245)),
             (( 280, 245), ( 750, 245)),
             (( 280, 245), ( 400, 245)),
             (( 400, 245), ( 520, 245)),
             (( 520, 245), ( 640, 245)),
             (( 640, 245), ( 750, 245)),
             (( 750, 245), ( 900, 245)),
             (( 900, 245), ( 1040, 255)),
             (( 140, 245), ( 280, 310)),
             (( 140, 310), ( 280, 245)),
             (( 0, 325), ( 140, 245)),
             (( 255, 245), ( 390, 230)),
             (( 255, 245), ( 450, 230)),
             (( 390, 230), ( 430, 30)),
             (( 450, 230), ( 500, 30))
             ])

G._sources = set([(0, 325), (0, 245)])
G._sinks = set([(430, 30), (500, 30), (1040, 255), (1040, 340), (570, 740), (635, 740)])

# vertical direction
G.add_edges([
             # outer lane straight
             (( 430, 740), (430, 670)),
             (( 430, 670), ( 430, 120)),
             (( 430, 120), ( 430, 30))
             ])
G._sources.add((430, 740))
G.add_edges([
             # inner lane straight
             (( 500, 740), (500, 670)),
             (( 500, 670), (500, 380)),
             (( 500, 380), (500, 120)),
             (( 500, 380), (750, 310)),
             (( 500, 380), (750, 250)),
             (( 500, 120), (500, 30))
             ])
G._sources.add(( 500, 740))


def plot_edges(graph, plt_src_snk = False):
    for start_node in G._edges:
        start_x = start_node[0]
        start_y = start_node[1]
        for end_node in G._edges[start_node]:
            end_x = end_node[0]
            end_y = end_node[1]
            dx = end_x - start_x
            dy = end_y - start_y
            # plot transition
            plt.arrow(start_x,start_y, dx, dy, linestyle='dashed', color = 'w',
                    width=edge_width,head_width=head_width, alpha=0.5)
    if plt_src_snk == True:
        node_x = np.zeros(len(graph._sources))
        node_y = np.zeros(len(graph._sources))
        k = 0
        for node in graph._sources:
            node_x[k] = node[0]
            node_y[k] = node[1]
            k += 1
        plt.plot(node_x, node_y, 'ro', markersize=10)
        node_x = np.zeros(len(graph._sinks))
        node_y = np.zeros(len(graph._sinks))
        k = 0
        for node in graph._sinks:
            node_x[k] = node[0]
            node_y[k] = node[1]
            k += 1
        plt.plot(node_x, node_y, 'bo', markersize=10)
        plt.legend(['sources', 'sinks'])

plot_edges(G, True)
G.print_graph()
plt.show()
