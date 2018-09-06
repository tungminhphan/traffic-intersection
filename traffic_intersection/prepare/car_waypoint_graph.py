# Waypoint Graphs for Primitive Computation
# Tung M. Phan
# California Institute of Technology
# May 9th, 2018
if __name__ == '__main__':
    import sys
    one_up = sys.path[0] + '/..'
    sys.path.append(one_up)
    visualize = True
else:
    # if this file is not called directly, don't plot
    visualize = False
import os
import numpy as np
from prepare.graph import WeightedDirectedGraph
from primitives.load_primitives import get_prim_data
import primitives.load_primitives as load_primitives

G = WeightedDirectedGraph()
for prim_id in range(load_primitives.num_of_prims):
    controller_found = get_prim_data(prim_id, 'controller_found')[0]
    if controller_found:
        from_node = tuple(get_prim_data(prim_id, 'x0'))[2:4]
        to_node = tuple(get_prim_data(prim_id, 'x_f'))[2:4]
        new_edge = (from_node, to_node)
        G.add_edges([new_edge])


def add_source_sink(edge):
    start, end = edge
    # check source
    x = start[0]
    y = start[1]
    has_source = x < 30 or x > 1020 or y > 700 or y < 50
    x = end[0]
    y = end[1]

G._sources = set([(0, 325), (0, 245), (430, 740), (500, 740)])
G._sinks = set([(427, 22), (500, 30), (1040, 255), (1040, 340), (570, 740), (635, 740)])

# temporary sources and sinks
offset_x = -82
offset_y = 2
G._sources.add((1144+offset_x, 515+offset_y))
G._sources.add((1144+offset_x, 435+offset_y))
G._sources.add((714+offset_x, 20+offset_y))
G._sources.add((644+offset_x, 20+offset_y))
G._sinks.add((714+offset_x, 730+offset_y))
G._sinks.add((644+offset_x, 730+offset_y))
G._sinks.add((104+offset_x, 505+offset_y))
G._sinks.add((104+offset_x, 420+offset_y))
A = np.array([570+offset_x, 740+offset_y])
B = np.array([644+offset_x, 730+10])

print('all the sinks are')
print(G._sinks)
print('all the sources are')
print(G._sources)
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
    print('the size of the background is')
    xmax, ymax = background.size
    origin = np.array([[xmax],[ymax]])/2.
    source_old = np.array([[0],[325]])
    source_new = origin+origin-source_old

    source_wrong = np.array([[1144], [435]])
    print(source_new-source_wrong)
    plt.show()
