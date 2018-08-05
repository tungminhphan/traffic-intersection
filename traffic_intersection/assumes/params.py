# figure params
length = 788
car_width = 399
axle_to_back = 160
center_to_axle_dist = length/2. - axle_to_back
front_to_axle = length-axle_to_back
pedestrian_scale_factor = 0.32
car_scale_factor = 0.1 # scale for when L = 50?
num_subprims = 5
theta_compensate = 5

pixel_to_meter_scale_factor = 50 / 1.5 # 1.5 meters ~ 50 pixels

# constants
g = 9.81 # gravitational constant
#g = pixel_to_meter_scale_factor * 9.81 # gravitational constant
