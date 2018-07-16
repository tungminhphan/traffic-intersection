# Collision Detection Cars and Pedestrians
# Anhminh Nguyen
# July 10, 2018

from math import cos, sin
import numpy as np

#input center coords of car to get its unrotated vertices
def vertices_car(x, y):
    return [(x-47,y-24), (x-47,y+24), (x+47,y+24), (x+47,y-24)]

#***********GET THE DIMENSIONS OF PEDESTRIAN**********
def vertices_pedestrian(x, y):
    return [(x-47,y-24), (x-47,y+24), (x+47,y+24), (x+47,y-24)]

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
    return [edge_vector(vertices[i], vertices[(i + 1)]) for i in range(2)]

#gets the normal of surface/edges
def get_axis(v):
    return (v[1], -v[0])

#dots all of shape's vertices with axis then returns the min and max respectively in a list
def projection(vertices, axis):
    projections = [dot(vertex, axis) for vertex in vertices]
    return [min(projections), max(projections)]

#all cases must be true/overlap for collision, (min, max)
def overlap(s1, s2):
    if (s1[0] >= s2[0]) and (s1[0] <= s2[1]):
        return True
    elif (s1[1] >= s2[0]) and (s1[1] <= s2[1]):
        return True
    elif (s2[0] >= s1[0]) and (s2[0] <= s1[1]):
        return True
    elif (s2[1] >= s1[0]) and (s2[1] <= s1[1]):
        return True
    else:
        return False

#takes two objects (pedestrians only atm) and checks if they are colliding
def collision_check(object1, object2):
    x, y, theta, gait = object1.state
    vertices_a = vertices_car(x, y)
    object1_vertices = [rotate_vertex(x, y, theta, vertex) for vertex in vertices_a]

    x2, y2, theta2, gait2 = object2.state
    vertices_b = vertices_car(x2, y2)
    object2_vertices = [rotate_vertex(x2, y2, theta2, vertex) for vertex in vertices_b]

    edges_a = vectors_of_edges(object1_vertices) #list of the vectors of the edges/sides  
    edges_b = vectors_of_edges(object2_vertices)
    edges = edges_a + edges_b

    axes = [get_axis(edge) for edge in edges]
    
    for i in range(4):
        projection_a = projection(object1_vertices, axes[i]) # (min, max)
        projection_b = projection(object2_vertices, axes[i])
        overlapping = overlap(projection_a, projection_b)
        if not overlapping:
            return False 
    return True

