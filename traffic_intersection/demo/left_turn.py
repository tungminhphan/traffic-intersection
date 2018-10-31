#!/usr/local/bin/python
# Left Turn Scenario
# Tung M. Phan
# California Institute of Technology
# October 29, 2018
import sys
sys.path.append('..')
from prepare.helper import *
import time, platform, warnings, matplotlib, random
import components.scheduler as scheduler
import datetime
if platform.system() == 'Darwin': # if the operating system is MacOS
    matplotlib.use('macosx')
else: # if the operating system is Linux or Windows
    try:
        import PySide2 # if pyside2 is installed
        matplotlib.use('Qt5Agg')
        matplotlib.use('TkAgg') # this may be slower
    except ImportError:
        warnings.warn('Using the TkAgg backend, this may affect performance. Consider installing pyside2 for Qt5Agg backend')
        matplotlib.use('TkAgg') # this may be slower
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from PIL import ImageFont, ImageDraw

# creates figure
fig = plt.figure()
ax = fig.add_axes([0,0,1,1]) # get rid of white border
if not options.show_axes:
    plt.axis('off')
# draw background
background = intersection.get_background()
# define draw
draw = ImageDraw.Draw(background)

# draw traffic lights
traffic_lights = traffic_signals.TrafficLights(yellow_max = 10, green_max = 50, random_start = False, horizontal_state = ['green', 28])
update_traffic_lights(ax, plt, traffic_lights) # for plotting

# draw_cars
car1 = car.KinematicCar(init_state=(0,0,250,380), color='red1')
car2 = car.KinematicCar(init_state=(0,0,250,520), color='green')
car3 = car.KinematicCar(init_state=(0,np.pi,550,310), color='yellow')
car4 = car.KinematicCar(init_state=(0,np.pi,900,240), color='blue')
cars = [car1, car2, car3, car4]
draw_cars(cars, background)

# draw_pedestrians
human1 = pedestrian.Pedestrian(init_state=(20,20,20,20))
humans = [human1]
draw_pedestrians(humans, background)

# draw detected
fontsize = 50
font = ImageFont.truetype("NotoSans-BoldItalic.ttf", fontsize)
draw.text((550, 310), '!', font=font)
draw.text((900, 240), '!', font=font)

# draw field of view
theta = car1.state[1]
theta_width = 50
depth_of_field = 1000
rear_axle_to_eye = 30
offset = [rear_axle_to_eye*np.cos(theta), rear_axle_to_eye*np.sin(theta)]
patches = [Wedge((car1.state[2]+offset[0], car1.state[3])+offset[1], depth_of_field, theta-theta_width, theta+theta_width)]
p = PatchCollection(patches, alpha=0.50)
ax.add_collection(p)
# show background
ax.imshow(background)
plt.show()


