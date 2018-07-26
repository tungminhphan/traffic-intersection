# Waypoint Graphs for Primitive Computation
# Tung M. Phan
# California Institute of Technology
# May 9th, 2018

import os
import numpy as np
visualize = True
if __name__ == '__main__':
    from graph import WeightedDirectedGraph
else:
    # if this file is not called directly, don't plot
    from prepare.graph import WeightedDirectedGraph
    visualize = False

G = WeightedDirectedGraph()
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


# vertical direction
G.add_edges([
             # outer lane straight
             (( 430, 740), (430, 670)),
             (( 430, 670), ( 430, 120)),
             (( 430, 120), ( 430, 30))
             ])
G.add_edges([
             # inner lane straight
             (( 500, 740), (500, 670)),
             (( 500, 670), (500, 380)),
             (( 500, 380), (500, 120)),
             (( 500, 380), (750, 310)),
             (( 500, 380), (750, 250)),
             (( 500, 120), (500, 30))
             ])
G._sources = set([(0, 325), (0, 245), (430, 740), (500, 740)])
G._sinks = set([(430, 30), (500, 30), (1040, 255), (1040, 340), (570, 740), (635, 740)])

# temporary sources and sinks
G._sources.add((1144-74, 515+10))
G._sources.add((1144-74, 435+10))
G._sources.add((714-74, 20+10))
G._sources.add((644-74, 20+10))
G._sinks.add((714-74, 730+10))
G._sinks.add((644-74, 730+10))
G._sinks.add((104-74, 505+10))

G._sinks.add((104-74, 420+10))

A = np.array([570-74, 740+10])
B = np.array([644-74, 730+10])

if visualize:
    import matplotlib.pyplot as plt
    from PIL import Image
    dir_path = os.path.dirname(os.path.realpath(__file__))
    intersection_fig = os.path.dirname(dir_path) + "/components/imglib/intersection.png"
    fig = plt.figure()
    plt.axis("on") # turn on/off axes
    background = Image.open(intersection_fig)
    xlim, ylim = background.size
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    plt.imshow(background, origin="Lower")
    markersize = 1
    edge_width = 3
    head_width = 20
    G.plot_edges(plt, plt_src_snk=True, edge_width = edge_width, head_width = head_width, markersize=markersize)
    G.print_graph()
    plt.show()
