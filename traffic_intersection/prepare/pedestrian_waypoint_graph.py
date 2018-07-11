# Waypoint Graphs for Pedestrian Automaton Computation
# Tung M. Phan
# California Institute of Technology
# July 10th, 2018

import os
import numpy as np
if __name__ == '__main__':
    from graph import WeightedDirectedGraph
else:
    from traffic_intersection.prepare.graph import WeightedDirectedGraph
import matplotlib.pyplot as plt
from PIL import Image
dir_path = os.path.dirname(os.path.realpath(__file__))
intersection_fig = os.path.dirname(dir_path) + "/components/imglib/intersection.png"
fig = plt.figure()

# turn on/off axes
plt.axis("on")
background = Image.open(intersection_fig)
xlim, ylim = background.size
plt.xlim(0, xlim)
plt.ylim(0, ylim)

plt.imshow(background, origin="Lower")

markersize = 30
edge_width = 0.5
head_width = 10
transparency = 0.5

G = WeightedDirectedGraph()
# define sources/sinks
left_bottom = (0, 170)
right_bottom = (1062, 170)
left_top = (0, 590)
right_top = (1062, 590)

top_left = (355, 762)
top_right = (705, 762)
bottom_left = (355, 0)
bottom_right = (705, 0)

G._sources = set([left_bottom]) #add left-bottom
G._sources.add(right_bottom) #add right-bottom
G._sources.add(left_top) #add left-top
G._sources.add(right_top) #add right-top

G._sources.add(top_left) #add top-left
G._sources.add(top_right) #add top-right
G._sources.add(bottom_left) #add bottom-left
G._sources.add(bottom_right) #add bottom-right

G._sinks = G._sources.copy() # sinks are the same nodes as sources

offset_wait = 25 # distance from the "pedestrian intersection" to waiting location
wait_top_left = (355, 590)
wait_top_left_vertical = (355, 590-offset_wait)
wait_top_left_horizontal = (355+offset_wait, 590)

wait_bottom_left = (355, 170)
wait_bottom_left_vertical = (355, 170+offset_wait)
wait_bottom_left_horizontal = (355+offset_wait, 170)

wait_top_right = (705, 590)
wait_top_right_vertical = (705, 590-offset_wait)
wait_top_right_horizontal = (705-offset_wait, 590)

wait_bottom_right = (705, 170)
wait_bottom_right_vertical = (705, 170+offset_wait)
wait_bottom_right_horizontal = (705-offset_wait, 170)

# for visualization only (not real sinks) TODO: delete
#G._sinks.add(wait_top_left)
#G._sinks.add(wait_top_left_vertical)
#G._sinks.add(wait_top_left_horizontal)
#
#G._sinks.add(wait_bottom_left)
#G._sinks.add(wait_bottom_left_vertical)
#G._sinks.add(wait_bottom_left_horizontal)
#
#G._sinks.add(wait_top_right)
#G._sinks.add(wait_top_right_vertical)
#G._sinks.add(wait_top_right_horizontal)
#
#G._sinks.add(wait_bottom_right)
#G._sinks.add(wait_bottom_right_vertical)
#G._sinks.add(wait_bottom_right_horizontal)

# add edges
all_edges = [
             (left_bottom, wait_bottom_left),
             (bottom_left, wait_bottom_left),
             (wait_bottom_left_horizontal, wait_bottom_left),
             (wait_bottom_left_vertical, wait_bottom_left),

             (right_bottom, wait_bottom_right),
             (bottom_right, wait_bottom_right),
             (wait_bottom_right_horizontal, wait_bottom_right),
             (wait_bottom_right_vertical, wait_bottom_right),

             (left_top, wait_top_left),
             (top_left, wait_top_left),
             (wait_top_left_horizontal, wait_top_left),
             (wait_top_left_vertical, wait_top_left),

             (right_top, wait_top_right),
             (top_right, wait_top_right),
             (wait_top_right_horizontal, wait_top_right),
             (wait_top_right_vertical, wait_top_right),

             (wait_bottom_left_vertical, wait_top_left_vertical),
             (wait_bottom_left_horizontal, wait_bottom_right_horizontal),
             (wait_top_right_vertical, wait_bottom_right_vertical),
             (wait_top_right_horizontal, wait_top_left_horizontal)
        ]

# add weights as physical distances
for edge in all_edges:
    node_1 = edge[0]
    node_2 = edge[1]
    weight = np.linalg.norm(np.array(edge[0]) - np.array(edge[1]))
    G.add_double_edges([(node_1, node_2, weight)])

G.print_graph()
G.plot_edges(plt, alpha = transparency, edge_width = edge_width, head_width=head_width,
        plt_src_snk=False)
plt.show()
