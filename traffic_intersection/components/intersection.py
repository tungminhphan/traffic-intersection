# The Intersection Component
# Tung M. Phan
# California Institute of Technology
# August 6, 2018
#
import os
from PIL import Image
dir_path = os.path.dirname(os.path.realpath(__file__))
#intersection_fig = dir_path + "/components/imglib/intersection_states/intersection_lights.png"
intersection_fig = os.path.dirname(dir_path) + '/components/imglib/intersection_states/intersection_lights.png'
intersection = Image.open(intersection_fig)

def get_background():
    return Image.open(intersection_fig)

def reflect(coords, limit):
    new_coords = []
    for x in coords:
        new_coords.append(limit - x)
    return new_coords

max_x, max_y = intersection.size
traffic_light_walls = dict()
traffic_light_walls['west'] ={'x': [345, 365, 365, 345], 'y': [415, 415, 210, 210]}
traffic_light_walls['east'] ={'x': reflect([345, 365, 365, 345], max_x), 'y': reflect([415, 415, 210, 210], max_y)}
traffic_light_walls['north'] ={'x': [395, 530, 530, 395], 'y': [600, 600, 580, 580]}
traffic_light_walls['south'] ={'x': reflect([395, 530, 530, 395], max_x), 'y': reflect([600, 600, 580, 580], max_y)}

crossing_walls = dict()
crossing_walls ['west'] ={'x': [344, 366, 366, 344], 'y': [553, 553, 209, 209]}
crossing_walls ['east'] ={'x': reflect([344, 366, 366, 344], max_x), 'y': reflect([553, 553, 209, 209], max_y)}
crossing_walls ['north'] ={'x': [395, 670, 670, 395], 'y': [600, 600, 580, 580]}
crossing_walls['south'] ={'x': reflect([395, 670, 670, 395], max_x), 'y': reflect([600, 600, 580, 580], max_y)}

##################################
#                                #
#          VISUALIZATION         # 
#                                #
##################################
visualize = False
if visualize:
    import matplotlib.pyplot as plt
    dir = 'south'
    plt.imshow(intersection)
    xs = crossing_walls[dir]['x']
    xs.append(xs[0])
    ys = crossing_walls[dir]['y']
    ys.append(ys[0])
    plt.plot(xs,ys, 'r')
    plt.show()
