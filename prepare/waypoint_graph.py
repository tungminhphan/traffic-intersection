# Waypoint Graphs for Primitive Computation
# Tung M. Phan
# California Institute of Technology
# May 9th, 2018

import os
from numpy import cos, sin, pi
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
# a node is of the form ((v values), theta, x, y)
G.add_edges([
             # first lane
             (((0, 50), 0, 0, 325), ((0,50), 0, 140, 380)),
             (((0,50), 0, 140, 380), ((0,50), 0, 255, 380)),
             (((0,50), 0, 255, 380), ((0,50), 0, 400, 380)),
             (((0,50), 0, 400, 380), ((50), pi/2, 570, 610)),
             (((50), pi/2, 570, 610), (50, pi/2, 570, 740)),
             (((0,50), 0, 400, 380), ((50), pi/2, 635, 610)),
             (((50), pi/2, 635, 610), (50, pi/2, 635, 740)),
             # second lane
             (((0,50), 0, 0, 325), ((0,50), 0, 140, 310)),
             (((0,50), 0, 140, 310), ((0,50), 0, 255, 310)),
             (((0,50), 0, 255, 310), ((50), 0, 900, 310)),
             (((50), 0, 900, 310), ((50), 0, 1040, 340)),
             # third lane
             (((0,50), 0, 0, 245), ((0,50), 0, 140, 310)),
             (((0,50), 0, 0, 245), ((0,50), 0, 140, 245)),
             (((0,50), 0, 140, 245), ((0,50), 0, 255, 245)),
             (((0,50), 0, 255, 245), ((50), 0, 900, 245)),
             (((50), 0, 900, 245), ((0,50), 0, 1040, 255)),
             (((0,50), 0, 140, 245), ((0,50), 0, 255, 310)),
             (((0,50), 0, 140, 310), ((0,50), 0, 255, 245)),
             (((0,50), 0, 0, 325), ((0,50), 0, 140, 245)),
             (((0,50), 0, 255, 245), ((50), -pi/4, 390, 230)),
             (((0,50), 0, 255, 245), ((50), -pi/4, 450, 230)),
             (((50), -pi/4, 390, 230), ((50), -pi/2, 430, 30)),
             (((50), -pi/4, 450, 230), ((50), -pi/2, 500, 30))
             ])

for start_node in G._edges:
    start_x = start_node[2]
    start_y = start_node[3]
    plot_car(start_node)
    for end_node in G._edges[start_node]:
        end_x = end_node[2]
        end_y = end_node[3]
        dx = end_x - start_x
        dy = end_y - start_y
        plot_car(end_node)
        # plot transition
        plt.arrow(start_x,start_y, dx, dy, linestyle='dashed', color = 'r',
                width=edge_width,head_width=head_width, alpha=0.5)

G.print_graph()
plt.show()
