# Collision Detection Cars and Pedestrians
# Anhminh Nguyen, Tung M. Phan
# July 10, 2018

import os, sys
sys.path.append("..")
from math import cos, sin
from components.pedestrian import Pedestrian
from components.car import KinematicCar
import assumes.params as params

# input center coords of car to get its unrotated vertices
def vertices_car(x, y):
    # x, y are the coordinates of the center
    # half the width and height of scaled car
    w = 788 * params.car_scale_factor / 2
    h = 399 * params.car_scale_factor / 2
    return [(x - w, y - h), (x - w, y + h), (x + w, y + h), (x + w, y - h)]

# diamond-like vertices
def vertices_pedestrian(x, y):
    w1 = 16 * params.pedestrian_scale_factor
    w2 = 27 * params.pedestrian_scale_factor
    h1 = 35 * params.pedestrian_scale_factor
    h2 = 35 * params.pedestrian_scale_factor
    return [(x - w1, y), (x, y + h1), (x + w2, y), (x, y - h2)]

# rotates the vertices based on car/pedestrian orientation
def rotate_vertex(x, y, theta, v):
    return ((v[0] - x) * cos(theta) - (v[1] - y) * sin(theta) + x, (v[0] - x) * sin(theta) + (v[1] - y) * cos(theta) + y)

# used for projection of axes and vertices
def dot(v1, v2):
	return v1[0] * v2[0] + v1[1] * v2[1]

# gets the vector between two vertices (the edge)  
def edge_vector(vertex1, vertex2):
	return (vertex2[0] - vertex1[0], vertex2[1] - vertex1[1])

# takes the vectors of the edges returns them in list, note that the input list of vertices must be given in an order such that following them will trace out the polygon
def vectors_of_edges(vertices):
    return [edge_vector(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]

#gets the perpendicular/normal of the edges
def get_axis(v):
    return (v[1], -v[0])

#dots all of shape's vertices with one of the separating axis then returns the min and max value respectively
def projection(vertices, axis):
    projections = [dot(vertex, axis) for vertex in vertices]
    return [min(projections), max(projections)]

#checks if there's overlap of two invervals s1 and s2
#s1 holds (min, max) of object 1
#s2 holds (min, max) of object 2
def overlap(s1, s2):
    return not (s1[1] < s2[0] or s2[1] < s1[0])

# if distances of centers are greater than sum of radii then for sure no collision
# this function returns False is there may be a collision, True when collision can't possibly happen
def no_collision_by_radius_check(x1, y1, r1, x2, y2, r2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 > (r1 + r2)

#takes two objects and checks if they are colliding
def get_bounding_box(thing):
    if type(thing) is Pedestrian:
        x, y, theta, gait = thing.state
        vertices = vertices_pedestrian(x, y)
        radius = 40 * params.pedestrian_scale_factor #the longest distance used for quick circular bounding box
    elif type(thing) is KinematicCar:
        vee, theta, x, y = thing.state
        r = params.center_to_axle_dist*params.car_scale_factor
        x, y = x+r*cos(theta), y + r*sin(theta)
        vertices = vertices_car(x, y)
        radius = ((788 * params.car_scale_factor / 2) ** 2 + (399 * params.car_scale_factor / 2) ** 2) ** 0.5
    else:
        TypeError('Not sure what this object is')

    rotated_vertices = [rotate_vertex(x, y, theta, vertex) for vertex in vertices]
    return rotated_vertices, x, y, radius

def collision_free(object1, object2):
    #this function returns True if no collision has happened, False otherwise 
    object1_vertices, x, y, radius = get_bounding_box(object1)
    object2_vertices, x2, y2, radius2 = get_bounding_box(object2)

    #takes the distance of the centers and compares it to the sum of radius, if the distance is greater then collision not possible
    if no_collision_by_radius_check(x, y, radius, x2, y2, radius2):
        return True
    else: # deep collision check
        #concatenate lists of the vectors of the edges/sides
        edges = vectors_of_edges(object1_vertices) + vectors_of_edges(object2_vertices)

        #gets the vectors perpendicular to the edges
        axes = [get_axis(edge) for edge in edges]

        all_overlapping = True # assume this is True initially, we will check if this is actually the case
        # look for overlapping in projections to each axis
        for i in range(len(axes)):
            projection_a = projection(object1_vertices, axes[i])#dots all of the vertices with one of the separating axis and returns the (min, max) projections 
            projection_b = projection(object2_vertices, axes[i])
            overlapping = overlap(projection_a, projection_b) # (min,max) of both objects are compared for overlap
            all_overlapping = all_overlapping and overlapping # check if all_overlapping is still True
            if all_overlapping == False: # if the assumption that all intervals are overlapping turns out to be False, there is no collision since a separating hyperplane exists
                return True
        return False
