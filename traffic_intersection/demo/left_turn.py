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
    except ImportError:
        warnings.warn('Using the TkAgg backend, this may affect performance. Consider installing pyside2 for Qt5Agg backend')
        matplotlib.use('TkAgg') # this may be slower
import matplotlib.animation as animation
import matplotlib.pyplot as plt


#import numpy as np
#import matplotlib
#from matplotlib.patches import Wedge
#from matplotlib.collections import PatchCollection
#import matplotlib.pyplot as plt
#
#
#fig = plt.figure()
#ax = fig.add_axes([0,0,1,1]) # get rid of white border
#
## Some limiting conditions on Wedge
#theta = 50
#theta_width = 20
#depth_of_field = 0.5
#patches = [Wedge((0, 0), depth_of_field, theta-theta_width, theta+theta_width)]
#p = PatchCollection(patches, alpha=0.50)
#ax.add_collection(p)
#
#plt.show()


# set randomness
if not options.random_simulation:
    random.seed(options.random_seed)
    np.random.seed(options.np_random_seed )
# creates figure
fig = plt.figure()
ax = fig.add_axes([0,0,1,1]) # get rid of white border
if not options.show_axes:
    plt.axis('off')
# sampling time
dt = options.dt

# create traffic intersection
traffic_lights = traffic_signals.TrafficLights(yellow_max = 10, green_max = 50, random_start = True)

def animate(frame_idx): # update animation by dt
    t0 = time.time()
    global_vars.current_time = frame_idx * dt # update current time from frame index and dt

    # update traffic lights
    traffic_lights.update(dt)
    update_traffic_lights(ax, plt, traffic_lights) # for plotting
    _,_,the_car = spawn_car()
    draw_cars_fast(ax, [the_car])
    the_intersection = [ax.imshow(intersection.intersection, origin="lower")] # update the stage

    all_artists = the_intersection + global_vars.crossing_highlights + global_vars.honk_waves + global_vars.boxes + global_vars.curr_tubes + global_vars.ids + global_vars.prim_ids_to_show + global_vars.walls + global_vars.show_traffic_lights + global_vars.cars_to_show + global_vars.pedestrians_to_show + global_vars.walk_signs
    t1 = time.time()
    elapsed_time = (t1 - t0)
    print('{:.2f}'.format(global_vars.current_time)+'/'+str(options.duration) + ' at ' + str(int(1/elapsed_time)) + ' fps') # print out current time to 2 decimal places
    return all_artists
t0 = time.time()
animate(0)
t1 = time.time()
interval = (t1 - t0)
ani = animation.FuncAnimation(fig, animate, frames=int(options.duration/options.dt), interval=interval, blit=True, repeat=False) # by default the animation function loops so set repeat to False in order to limit the number of frames generated to num_frames
if options.save_video:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps = options.speed_up_factor*int(1/options.dt), metadata=dict(artist='Traffic Intersection Simulator'), bitrate=-1)
    now = str(datetime.datetime.now())
    ani.save('../movies/' + now + '.avi', dpi=200)
plt.show()
t2 = time.time()
print('Total elapsed time: ' + str(t2-t0))
