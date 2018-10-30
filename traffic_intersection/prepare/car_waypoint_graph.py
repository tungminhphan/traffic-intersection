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

ignore_set = {29,30,31,32, 129, 130, 131, 152, 153, 154, 155, 53, 54, 55, 56, 87, 88, 89, 90, 91, 92, 93}

def add_source_sink(edge):
    start, end, weight = edge
    # check source
    x = start[2]
    y = start[3]
    has_source = x < 50 or x > 1000 or y > 700 or y < 50
    if has_source:
        G._sources.add(start)
    x = end[2]
    y = end[3]
    has_sink = x < 50 or x > 1000 or y > 700 or y < 50
    if has_sink:
        G._sinks.add(end)

for prim_id in range(load_primitives.num_of_prims):
    controller_found = get_prim_data(prim_id, 'controller_found')[0]
    if controller_found and prim_id not in ignore_set:
        from_node = tuple(get_prim_data(prim_id, 'x0'))
        to_node = tuple(get_prim_data(prim_id, 'x_f'))
        time_weight = get_prim_data(prim_id, 't_end')[0]
        new_edge = (from_node, to_node, time_weight)
        label_set = [prim_id]
        G.add_edges([new_edge], use_euclidean_weight = False, label_edges = True, edge_label_set = label_set)
        add_source_sink(new_edge)

if visualize:
    import matplotlib.pyplot as plt
    from PIL import Image
    dir_path = os.path.dirname(os.path.realpath(__file__))
    intersection_fig = os.path.dirname(dir_path) + "/components/imglib/intersection_states/intersection_lights.png"
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
    G.plot_edges(plt, plt_src_snk = True, plt_labels = True, edge_width = edge_width, head_width = head_width, markersize=markersize)
    plt.show()
