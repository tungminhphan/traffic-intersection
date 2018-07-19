# Collision Detection Cars and Pedestrians
# Anhminh Nguyen
# July 10, 2018

from math import cos, sin
import numpy as np
from traffic_intersection.components.pedestrian import *

#input center coords of car to get its unrotated vertices
def vertices_car(x, y, car_scale_factor):
    #half the width and height of scaled car
    w = 788 * car_scale_factor / 2 
    h = 399 * car_scale_factor / 2
    return [(x - w, y - h), (x - w, y + h), (x + w, y + h), (x + w, y - h)]

#***********GET THE DIMENSIONS OF PEDESTRIAN**********
def vertices_pedestrian(x, y, pedestrian_scale_factor):
    w = 67 * pedestrian_scale_factor / 2 
    h = 50 * pedestrian_scale_factor / 2
    return [(x - w, y - h), (x - w, y + h), (x + w, y + h), (x + w, y - h)]

#get the rotated vertices based on car/pedestrian orientation
def rotate_vertex(x, y, theta, v): 
    return ((v[0] - x) * cos(theta) - (v[1] - y) * sin(theta) + x, \
(v[0] - x) * sin(theta) + (v[1] - y) * cos(theta) + y)


def dot(v1, v2):
	return v1[0] * v2[0] + v1[1] * v2[1]

#gets the vector between two points (the edges)  
def edge_vector(vertex1, vertex2):
	return (vertex2[0] - vertex1[0], vertex2[1] - vertex1[1])

#takes the vectors of the edges returns them in list
def vectors_of_edges(vertices):
    return [edge_vector(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]

#gets the normal of surface/edges
def get_axis(v):
    return (v[1], -v[0])

#dots all of shape's vertices with axis then returns the min and max respectively in a list
def projection(vertices, axis):
    projections = [dot(vertex, axis) for vertex in vertices]
    return [min(projections), max(projections)]

#checks if there's overlap
def helper(s1, s2):
    a = s2[0]
    b = s2[1]
    if b < a:
        a = s2[1]
        b = s2[0]
    return (s1 >= a) and (s1 <= b)

#all cases must be true/overlap for collision, (min, max)
def overlap(s1, s2):
    if helper(s1[0], s2):
        return True;
    if helper(s1[1], s2):
        return True;
    if helper(s2[0], s1):
        return True;
    if helper(s2[1], s1):
        return True
    return False

#if distances of centers are greater than sum of radii then no collision
def radius_check(x, y, width, x2, y2, width2):
    return ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5 > (width + width2)

#takes two objects and checks if they are colliding
def collision_check(object1, object2, car_scale_factor, pedestrian_scale_factor):
    if type(object1) == Pedestrian:
        x, y, theta, gait = object1.state
        vertices_a = vertices_pedestrian(x, y, pedestrian_scale_factor)
        radius = 67 * pedestrian_scale_factor
    else:
        vee, theta, x, y = object1.state
        vertices_a = vertices_car(x, y, car_scale_factor)
        radius = 788 * car_scale_factor
    
    if type(object2) == Pedestrian:
        x2, y2, theta2, gait2 = object2.state
        vertices_b = vertices_pedestrian(x2, y2, pedestrian_scale_factor)
        radius2 = 67 * pedestrian_scale_factor
    else:
        vee2, theta2, x2, y2 = object2.state
        vertices_b = vertices_car(x2, y2, car_scale_factor)
        radius2 = 788 * car_scale_factor

    #if distance of centers are greater than sum of radii then no collision
    if radius_check(x, y, radius, x2, y2, radius2):
        return False

    object1_vertices = [rotate_vertex(x, y, theta, vertex) for vertex in vertices_a]
    object2_vertices = [rotate_vertex(x2, y2, theta2, vertex) for vertex in vertices_b]

    edges_a = vectors_of_edges(object1_vertices) #list of the vectors of the edges/sides  
    edges_b = vectors_of_edges(object2_vertices)
    edges = edges_a + edges_b

    axes = [get_axis(edge) for edge in edges]
    
    for i in range(len(axes)):
        projection_a = projection(object1_vertices, axes[i]) # (min, max)
        projection_b = projection(object2_vertices, axes[i])
        overlapping = overlap(projection_a, projection_b)
        if not overlapping:
            return False 
    return True

