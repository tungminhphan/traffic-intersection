# Waypoint Graphs for Primitive Computation
# Tung M. Phan
# California Institute of Technology
# May 9th, 2018

import os
from graph import DirectedGraph
import matplotlib.pyplot as plt
from PIL import Image
dir_path = os.path.dirname(os.path.realpath(__file__))
intersection_fig = os.path.dirname(dir_path) + "/imglib/intersection.png"
fig = plt.figure()
# turn on/off axes
plt.axis("on")
background = Image.open(intersection_fig)
plt.imshow(background, origin="Lower")

markersize = 10
arrow_width = 4

G = DirectedGraph()
# a node is of the form (v, theta, x, y)
G.add_edges([((0, 0, 0, 315), (0, 0, 244,232)),
             ((0, 0, 200, 232), (0, 0, 442, 552))
             ])

for start_node in G._edges:
    start_x = start_node[2]
    start_y = start_node[3]
    plt.plot(start_x, start_y, 'r.', markersize = markersize)
    for end_node in G._edges[start_node]:
        end_x = end_node[2]
        end_y = end_node[3]
        dx = end_x - start_x
        dy = end_y - start_y
        plt.arrow(start_x,start_y, dx, dy, width = arrow_width, color = 'r')

plt.show()
