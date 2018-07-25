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
    w = 788 * params.car_scale_factor / 2.
    h = 399 * params.car_scale_factor / 2.
    return [(x - w, y - h), (x - w, y + h), (x + w, y + h), (x + w, y - h)]

# diamond-like vertices
def vertices_pedestrian(x, y):
    w1 = 27 * params.pedestrian_scale_factor
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

#returns centroid of polygon
def center_of_polygon(polygon_vertices):
    x = [v[0] for v in polygon_vertices]
    y = [v[1] for v in polygon_vertices]
    center = (sum(x) / len(polygon_vertices), sum(y) / len(polygon_vertices))
    return center

#checks if there's overlap of two invervals s1 and s2, returns the vector needed to separate the object along the specific axis
#s1 holds (min, max) of object 1
#s2 holds (min, max) of object 2
#axis is the separating axis
def overlap(s1, s2, axis):
    if (s1[1] < s2[0] or s2[1] < s1[0]): # boxes aren't overlapping, no separation vector 
        return False, None
    else: #return true and the vector needed to separate the objects along the axis
        d = min(s2[1] - s1[0], s1[1] - s2[0]) #(max2 - min1, max1 - min2) gets the distance of penetration
        factor = d / dot(axis, axis) #removes the scalar of axis from earlier projection between vertex and axis
        sv = (axis[0] * factor, axis[1] * factor) #separation vector
        return True, sv

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
        r = params.center_to_axle_dist * params.car_scale_factor
        x, y = x + r*cos(theta), y + r*sin(theta)
        vertices = vertices_car(x, y)
        radius = ((788 * params.car_scale_factor / 2) ** 2 + (399 * params.car_scale_factor / 2) ** 2) ** 0.5
    else:
        TypeError('Not sure what this object is')

    rotated_vertices = [rotate_vertex(x, y, theta, vertex) for vertex in vertices]
    return rotated_vertices, x, y, radius

def nonoverlapping_polygons(polygon1_vertices, polygon2_vertices): # SAT algorithm, returns smallest vector needed to separate polygons
    #concatenate lists of the vectors of the edges/sides
    edges = vectors_of_edges(polygon1_vertices) + vectors_of_edges(polygon2_vertices)

    #gets the vectors perpendicular to the edges
    axes = [get_axis(edge) for edge in edges]

    #stores the separation vectors if there's an overlap
    separation_vectors = []

    center1 = center_of_polygon(polygon1_vertices)
    center2 = center_of_polygon(polygon2_vertices)
    vector_of_centers = (center2[0] - center1[0], center2[1] - center1[1])

    all_overlapping = True # assume this is True initially, we will check if this is actually the case
    # look for overlapping in projections to each axis
    for i in range(len(axes)):
        projection_a = projection(polygon1_vertices, axes[i])#dots all of the vertices with one of the separating axis and returns the (min, max) projections 
        projection_b = projection(polygon2_vertices, axes[i])
        overlapping, sv = overlap(projection_a, projection_b, axes[i]) # (min,max) of both polygons are compared for overlap
        separation_vectors.append(sv) # adds the vector needed to separate the polygons along the axis
        all_overlapping = all_overlapping and overlapping # check if all_overlapping is still True
        if all_overlapping == False: # if the assumption that all intervals are overlapping turns out to be False, there is no collision since a separating hyperplane exists
            return True, None # if no collision, no separation vector

    min_sep_vector = min(separation_vectors, key = (lambda v: dot(v,v))) # gets the smallest vector needed to separate the objects
    if dot(min_sep_vector, vector_of_centers) > 0: # if vectors are the same direction, invert min_sep_vector, this is for consistency in finding contact points later
        min_sep_vector = (min_sep_vector[0] * -1, min_sep_vector[1] * -1) 

    return False, min_sep_vector # min_sep_vector is always opposite the direction of the vector from the center of object 1 to the center of object 2

def collision_free(object1, object2):
    #this function returns True if no collision has happened, False otherwise along with the min vector needed to separate the objects
    object1_vertices, x, y, radius = get_bounding_box(object1)
    object2_vertices, x2, y2, radius2 = get_bounding_box(object2)

    #takes the distance of the centers and compares it to the sum of radius, if the distance is greater then collision not possible
    if no_collision_by_radius_check(x, y, radius, x2, y2, radius2):
        return True, None
    else: # deep collision check
        return nonoverlapping_polygons(object1_vertices, object2_vertices)


################################ CONTACT POINTS ################################
'''
def closest_edge(polygon_vertices, separation_normal): # the closest edge is the edge most perpendicular to the separation normal
    max_idx, max_value = max(enumerate(polygon_vertices), key = lambda v: dot(v[1],separation_normal))
    if max_index == 0:
        next_idx 
        prev_idx 
    v = polygon_vertices[max_idx]
    v1 = 

def contact_points(object1, object2, separation_normal):
'''
