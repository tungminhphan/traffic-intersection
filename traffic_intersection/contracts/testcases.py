import automaton as a
import numpy as np

h_traffic = ['red_h', 'green_h', 'yellow_h']
h_trans = {}
h_trans[('red_h', 'green_h')] = ('h_timer == 45', set(), {'green_h'}, set())
h_trans[('green_h', 'yellow_h')] = ('h_timer == 40', set(), set(), set())
h_trans[('yellow_h', 'red_h')] = ('h_timer == 5', set(), {'red_h'}, set())
h_trans[('green_h', 'green_h')] = ('h_timer < 40', set(), {'green_h'}, set())
h_trans[('red_h', 'red_h')] = ('h_timer < 45', set(), {'red_h'}, set())
h_trans[('yellow_h', 'yellow_h')] = ('h_timer < 5', set(), set(), set())

h_traff_auto = a.construct_automaton(h_traffic, h_trans, 'red_h', {'red_h', 'green_h', 'yellow_h'})
h_traff_auto.convert_to_digraph().render('horizontal traffic lights', view = True)

v_traffic = ['red_v', 'green_v', 'yellow_v']
v_trans = {}
v_trans[('red_v', 'green_v')] = ('True', {'red_h'}, set(), set())
v_trans[('green_v', 'yellow_v')] = ('v_timer == 5', {'red_h'}, set(), set())
v_trans[('yellow_v', 'red_v')] = ('True', {'green_h'}, set(), set())

v_traff_auto = a.construct_automaton(v_traffic, v_trans, 'green_v', {'red_v', 'green_v', 'yellow_v'})
v_traff_auto.convert_to_digraph().render('vertical traffic lights', view = True)

road_states = ['full', 'not full', 'aux', 'start_road', 'aux2']
road_trans = {}
road_trans[('full', 'not full')] = ('g_exit', set(), {'can_exit'}, {'num_cars--'})
road_trans[('full', 'aux2')] = ('num_cars < max_cap', set(), set(), set())
road_trans[('aux2', 'not full')] = ('True', set(), set(), set())
road_trans[('start_road', 'full')] = ('num_cars == max_cap', set(), set(), set())
road_trans[('start_road', 'not full')] = ('num_cars < max_cap', set(), set(), set())
road_trans[('not full', 'aux')] = ('g_enter', set(), {'can_enter'}, {'num_cars++'})
road_trans[('aux', 'full')] = ('num_cars == max_cap', set(), set(), set())
road_trans[('aux', 'not full')] = ('num_cars < max_cap', set(), set(), set())
road_trans[('not full', 'not full')] = ('g_exit', set(), {'can_exit'}, {'num_cars--'})

road_auto = a.construct_automaton(road_states, road_trans, 'start_road', {'full', 'not full'})
road_auto.convert_to_digraph().render('road', view = True)

# still need to add possibility of multiple transitions between states
car_states = ['start_car', 'enter', 'moving', 'stop', 'exit']
car_trans = {}
car_trans[('start_car', 'enter')] = ('at_entrance', {'can_enter'}, set(), set())
car_trans[('enter', 'moving')] = ('safe', set(), set(), set())
car_trans[('moving', 'moving')] = ('safe_to_speed_up, v < speed_limit', set(), set(), {'speed_up'})
car_trans[('moving', 'stop')] = ('v == 0', set(), set(), set())
car_trans[('stop', 'moving')] = ('safe_to_speed_up', set(), set(), {'speed_up'})
car_trans[('moving', 'exit')] = ('at_exit', {'can exit'}, set(), set())

car_auto = a.construct_automaton(car_states, car_trans, 'start_car', {'exit'})

total = a.compose_multiple_components([h_traff_auto, v_traff_auto, road_auto, car_auto])
total.trim()
total.convert_to_digraph().render('road', view = True)

pedestrian_states = ['start_road', 'stopped_horizontal', 'crossed_horizontal', 'crossed_vertical', 'stopped_vertical', 'exit']



