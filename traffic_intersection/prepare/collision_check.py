# Collision Detection Cars and Pedestrians
# Anhminh Nguyen, Tung M. Phan
# July 10, 2018

import os, sys
sys.path.append("..")
from math import cos, sin
from components.pedestrian import Pedestrian
from components.car import KinematicCar, DynamicCar
import assumes.params as params
import numpy as np

# input center coords of car to get its unrotated vertices
def vertices_car(x, y):
    # x, y are the coordinates of the center
    # half the width and height of scaled car
    w = 788 * params.car_scale_factor / 2.
    h = 399 * params.car_scale_factor / 2.
    return [(x - w, y - h), (x + w, y - h), (x + w, y + h), (x - w, y + h)] # counter clockwise

# diamond-like vertices
def vertices_pedestrian(x, y):
    w1 = 27 * params.pedestrian_scale_factor
    w2 = 27 * params.pedestrian_scale_factor
    h1 = 35 * params.pedestrian_scale_factor
    h2 = 35 * params.pedestrian_scale_factor
    return [(x - w1, y), (x, y - h2), (x + w2, y), (x, y + h1)] #counter clockwise

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
def overlap(s1, s2, axis):
    if (s1[1] < s2[0] or s2[1] < s1[0]):
        return False, None
    else: #return true and the vector needed to separate the objects along the axis
        d = min(s2[1] - s1[0], s1[1] - s2[0]) #(max2 - min1, max1 - min2) gets the distance of of penetration
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
    # takes the rotated vertices and finds the leftmost bottom vertex, orders the vertices in the counter clockwise direction with the leftmost bottom being the first one in the list
    min_index, min_value = min(enumerate(rotated_vertices), key = lambda v: v[1])
    ordered_vertices = rotated_vertices[min_index:] + rotated_vertices[:min_index]
    return ordered_vertices, x, y, radius

def nonoverlapping_polygons(polygon1_vertices, polygon2_vertices): # SAT algorithm
    #concatenate lists of the vectors of the edges/sides
    edges = vectors_of_edges(polygon1_vertices) + vectors_of_edges(polygon2_vertices)

    #gets the vectors perpendicular to the edges
    axes = [get_axis(edge) for edge in edges]


    #stores the separation vectors if there's an overlap
    separation_vectors = []

    center1 = center_of_polygon(polygon1_vertices)
    center2 = center_of_polygon(polygon2_vertices)
    vector_of_centers = (center1[0] - center2[0], center1[1] - center2[1]) # if min_sep_vector pointing in same direction as this vector, used to invert direction min_sep_vector


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

    return False, min_sep_vector

def collision_free(object1, object2):
    #this function returns True if no collision has happened, False otherwise 
    object1_vertices, x, y, radius = get_bounding_box(object1)
    object2_vertices, x2, y2, radius2 = get_bounding_box(object2)

    #takes the distance of the centers and compares it to the sum of radius, if the distance is greater then collision not possible
    if no_collision_by_radius_check(x, y, radius, x2, y2, radius2):
        return True, None
    else: # deep collision check
        return nonoverlapping_polygons(object1_vertices, object2_vertices)


################################ CONTACT POINTS ################################

def normalize(v):
    norm = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return (v[0] / norm, v[1] / norm)

def invert_direction(v):
    return (v[0] * -1, v[1] * -1)

def best_edge(polygon, separation_normal): # the closest edge is the edge most perpendicular to the separation normal
    polygon_vertices,_,_,_ = get_bounding_box(polygon)
    max_proj = 0.0
    max_idx = 0
    for idx, v in enumerate(polygon_vertices):
        projection = dot(v, separation_normal)
        if (projection > max_proj):
            max_proj = projection
            max_idx = idx
# gets the prev and next index w.r.t. max index 
    if max_idx == 0:
        next_idx = max_idx + 1
        prev_idx = len(polygon_vertices) - 1
    elif max_idx == (len(polygon_vertices) - 1):
        next_idx = 0
        prev_idx = max_idx - 1
    else:
        next_idx = max_idx + 1
        prev_idx = max_idx - 1

    v = polygon_vertices[max_idx]
    v1 = polygon_vertices[next_idx]
    v0 = polygon_vertices[prev_idx]
    #print(v)

    # the edges next to the max_vertex, pointing towards the max vertex
    right_edge = (v[0] - v1[0], v[1] - v1[1])
    left_edge = (v[0] - v0[0], v[1] - v0[1])
    right_edge = normalize(right_edge)
    left_edge = normalize(left_edge)

    #returns the most perpendicular edge (normalized), max vertex, and first and second vertex counter clockwise
    if (dot(left_edge, separation_normal) <= dot(right_edge, separation_normal)):
        return left_edge, v, v, v0
    else:
        return invert_direction(right_edge), v, v1, v

def clip_points(v1, v2, n, o): # clips line segment points v1, v2 if they are past o along n
    cp = []
    d1 = dot(n, v1) - o
    d2 = dot(n, v2) - o

    if d1 >= 0.0: # if point is past o along n then the point is kept
        cp.append(v1)
    if d2 >= 0.0:
        cp.append(v2)
    if d1 * d2 < 0.0: # checks if the points are on opposing sides to compute the correct point
        e = (v2[0] - v1[0], v2[1] - v1[1]) # v2 - v1, gets the vector of edge thats being clipped
        u = d1 / (d1 - d2)
        e = (u * e[0], u * e[1])
        e = (e[0] + v1[0], e[1] + v1[1])
        cp.append(e)
    return cp

def contact_points(object1, object2, separation_normal):
    invert_normal = invert_direction(separation_normal) # keep consistent separation normal from object 2 to object 1
    edge1_comps = best_edge(object1, separation_normal)
    edge2_comps = best_edge(object2, invert_normal)

    flip = False # flag indicating that incident and reference edge were flipped, this is for final clip
    # edge1_comps[0], edge2_comps[0] are the best edge vectors of their shapes
    if abs(dot(edge1_comps[0], separation_normal)) <= abs(dot(edge2_comps[0], separation_normal)):
        ref_edge, ref_vmax, ref_v2, ref_v1 = edge1_comps
        inc_edge, inc_vmax, inc_v2, inc_v1 = edge2_comps
    else:
        flip = True
        ref_edge, ref_vmax, ref_v2, ref_v1 = edge2_comps
        inc_edge, inc_vmax, inc_v2, inc_v1 = edge1_comps

    ref_v_invert = invert_direction(ref_edge)

    offset1 = dot(ref_edge, ref_v1) # offset of first vertex along ref edge vector 
    cp = clip_points(inc_v1, inc_v2, ref_edge, offset1) # clips the incident edge by the first vertex of the reference edge
    if len(cp) < 2: # less than two points then it fails and returns 
        return cp

    offset2 = dot(ref_edge, ref_v2) # offset, second vertex along ref edge vector
    cp = clip_points(cp[0], cp[1], ref_v_invert, -offset2) # clips what's left of incident edge by the second vertex of the reference edge in the opposite direction, the offset and direction are flipped
    if len(cp) < 2:
        return cp

    ref_normal = invert_direction(get_axis(ref_edge)) # reference edge normal towards its own center
    # gets the largest depth and makes sure the points are not past this
    max1 = dot(ref_normal, ref_vmax)
    if (flip):
        if dot(ref_normal, cp[1]) - max1 > 0.0:
            del cp[1]
        if dot(ref_normal, cp[0]) - max1 > 0.0:
            del cp[0]
    else:
        if dot(ref_normal, cp[1]) - max1 < 0.0:
            del cp[1]
        if dot(ref_normal, cp[0]) - max1 < 0.0:
            del cp[0]
    return cp

######################### Collision Response ######################### 
# need to test, currently only for cars
def get_motion_data(thing):
    if type(thing) is Pedestrian:
        x,y,theta,vee = thing.state
        velocity = np.array([vee*cos(theta),vee*sin(theta)]) # pre-collision velocity of center of mass
        omega = 0 # angular velocity, yaw rate 
        mass = 100
        w = 54 * params.pedestrian_scale_factor
        h = 70 * params.pedestrian_scale_factor
        inertia = mass * (h ** 2 + w ** 2) / 24
    elif type(thing) is KinematicCar:
        vee,theta,x,y = thing.state
        velocity = np.array([vee*cos(theta),vee*sin(theta)]) # pre-collision velocity of center of mass
        omega = 0 # angular velocity, yaw rate
        mass = 100
        w = 788 * params.car_scale_factor
        h = 399 * params.car_scale_factor
        inertia = mass * (h ** 2 + w ** 2) / 12
    else:
        TypeError('Not sure what this object is')
    return x, y, velocity, omega, mass, inertia

def collision_response(object1, object2, cp, min_sep_vector): # returns the translational and rotational motion of the objects   
    x, y, v_a1, omega_a1, m_a, inert_a = get_motion_data(object1)
    x2, y2, v_b1, omega_b1, m_b, inert_b = get_motion_data(object2)

    e = 1.0 # coeff of restitution

    r_ap = np.array([cp[0][0] - x, cp[0][1] - y]) # vector from center obj1 towards contact point     
    r_bp = np.array([cp[0][0] - x2, cp[0][1] - y2]) # vector from center obj2 towards contact point 

    omega1_cross_rap = np.array([-omega_a1 * r_ap[1], omega_a1 * r_ap[0]]) # angular velocity cross r_ap
    omegb1_cross_rbp = np.array([-omega_b1 * r_bp[1], omega_b1 * r_bp[0]])

    v_ap1 = v_a1 + omega1_cross_rap # pre-collision velocities of the points of collision
    v_bp1 = v_b1 + omegb1_cross_rbp

    v_ab1 = v_ap1 - v_bp1 # relative velocity (pre-collision) v_ap1 - vbp1

    edge_obj2,_,_,_ = best_edge(object2, min_sep_vector) # returns best edge of object2
    n = get_axis(edge_obj2) # gets the normal from collision edge
    if dot(n, r_bp) < 0: # if opposite direction from collision point, invert normal
        n = invert_direction(n)
    n = np.array([n[0], n[1]])

    j = -(1 + e) * np.dot(v_ab1, n)
    j /= ((1 / m_a) + (1 / m_b) + (np.cross(r_ap, n))**2 / inert_a + (np.cross(r_bp, n))**2 / inert_b)

    # post collision translational motion
    v_a2 = v_a1 + j * n / m_a
    v_b2 = v_b1 - j * n / m_b

    # post collision rotational motion 
    omega_a2 = omega_a1 + np.cross(r_ap, j*n) / inert_a
    omega_b2 = omega_b1 - np.cross(r_bp, j*n) / inert_b

    return v_a2, omega_a2, v_b2, omega_b2
