# Traffic Lights Class
# Tung M. Phan
# May 31st, 2018
# California Institute of Technology

import random
import os
import sys
from PIL import Image
sys.path.append("..")
dir_path = os.path.dirname(os.path.dirname(os.path.realpath("__file__")))

vertical_light_coordinates = {'green':[(377, 642), (682,110)], 'yellow': [(377, 659), (682,126)], 'red': [(378, 675), (682.5, 144.5)]}
horizontal_light_coordinates = {'green':[(291, 193), (756, 566.25)], 'yellow': [(309, 193), (773, 566.25)], 'red': [(327, 193), (790, 566.25)]}

walk_sign_go = dir_path + '/components/imglib/go.png'
go_fig = Image.open(walk_sign_go).convert("RGBA")
go_fig = Image.open(walk_sign_go)
go_fig = go_fig.resize((18,18))

walk_sign_stop = dir_path + '/components/imglib/stop.png'
stop_fig = Image.open(walk_sign_stop).convert("RGBA")
stop_fig = Image.open(walk_sign_stop)
stop_fig = stop_fig.resize((18,18))

vertical_go_fig = go_fig.rotate(180, expand = False)
vertical_stop_fig = stop_fig.rotate(180, expand = False)
horizontal_go_fig = go_fig.rotate(-90, expand = True)
horizontal_stop_fig = stop_fig.rotate(-90, expand = True)

vertical_walk_fig = {True: vertical_go_fig, False: vertical_stop_fig}
horizontal_walk_fig = {True: horizontal_go_fig, False: horizontal_stop_fig}
walk_sign_figs = {'vertical': vertical_walk_fig, 'horizontal': horizontal_walk_fig}
walk_sign_coordinates = {'vertical': [(378, 617), (683, 87)], 'horizontal': [(269, 191), (732, 564)]}

class TrafficLights():
    def __init__(self, yellow_max = 5, green_max = 25, random_start = True, horizontal_state = ['red', 28], walk_sign_duration_factor = 3):
        '''
        @param yellow_max is the duration of yellow
        @param green_max is the duration of green
        @param random_start

        '''
        self._max_time = {'yellow': yellow_max, 'green': green_max, 'red': yellow_max + green_max}
        self._walk_sign_duration = self._max_time['green']/walk_sign_duration_factor
        if random_start:
            colors = ['red', 'yellow', 'green']
            random_color = random.choice(colors) 
            random_time = random.uniform(0, self._max_time[random_color])
            self._horizontal_init = [random_color, random_time]
            horizontal_state = self._horizontal_init
        else:
            self._horizontal_init = horizontal_state
            horizontal_state = self._horizontal_init
        vertical_state = self.get_counterpart(horizontal_state)
        self._state = {'horizontal': horizontal_state, 'vertical': vertical_state}

    def get_counterpart(self, state):
        color = state[0]
        time = state[1]

        if color == 'green':
            other_color = 'red'
            other_time = time
        elif color == 'yellow':
            other_color = 'red'
            other_time = self._max_time['green'] + time
        else: # color == 'red'
            if time <= 1: # both lights are red 
                other_color = 'red'
                other_time = time
            elif time // self._max_time['green'] >= 1:
                other_color = 'yellow'
                other_time = time - self._max_time['green']
            else:
                other_time = time
                other_color = 'green'
        return [other_color, other_time]

    def successor(self, color):
        if color == 'red':
            return 'green'
        elif color == 'green':
            return 'yellow'
        else:
            return 'red'

    def predict(self, dt, use_init_state = False):
        full_cycle = sum(self._max_time[k] for k in self._max_time)
        if use_init_state == True:
            current_color = self._horizontal_init[0]
            horizontal_time = (self._horizontal_init[1] + dt) % full_cycle # update and trim time
        else:
            horizontal_time = (self._state['horizontal'][1] + dt) % full_cycle # update and trim time
            current_color = self._state['horizontal'][0]
        next_color = self.successor(current_color)
        last_color = self.successor(next_color)
        if horizontal_time // self._max_time[current_color] < 1: 
            horizontal_color = current_color
        elif horizontal_time // (full_cycle - self._max_time[last_color]) < 1:
            horizontal_color = next_color
            horizontal_time = horizontal_time - self._max_time[current_color]
        else:
            horizontal_color = last_color
            horizontal_time = horizontal_time - self._max_time[current_color] - self._max_time[next_color]
        new_horizontal_state = [horizontal_color, horizontal_time]
        return new_horizontal_state
        self._state['horizontal'] = new_horizontal_state
        return new_horizontal_state

    def update(self, dt):
        new_horizontal_state = self.predict(dt)
        self._state['horizontal'] = new_horizontal_state
        self._state['vertical'] = self.get_counterpart(new_horizontal_state)

    def get_states(self, which_light, color_or_time):
        if color_or_time == 'color':
            color_or_time = 0
        else:
            color_or_time = 1
        return self._state[which_light][color_or_time]

    def get_elapsed_time(self, which):
        if which == 'horizontal':
            return self.get_states('horizontal', 'time')
        elif which == 'vertical':
            return self.get_states('vertical', 'time')

    

