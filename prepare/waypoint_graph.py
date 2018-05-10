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


#dots = plt.axes().plot(300,300,'.')
markersize = 1
arrow_width = 3

x = (0,315)

G = DirectedGraph()

G.add_edges([(,
             (1,2),
             ])
G.print_graph()

plt.plot(x[0],x[1], 'r.', markersize = markersize)
plt.arrow(x[0],x[1], 20, 60, width = arrow_width, color = 'r')
plt.show()
