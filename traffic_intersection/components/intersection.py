# The Intersection Component
# Tung M. Phan
# California Institute of Technology
# August 6, 2018
#
import os
from PIL import Image
dir_path = os.path.dirname(os.path.realpath(__file__))
intersection_fig = os.path.dirname(dir_path) + "/components/imglib/intersection.png"
intersection = Image.open(intersection_fig)
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

