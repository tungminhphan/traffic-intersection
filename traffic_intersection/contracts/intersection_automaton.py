from contract_automaton import *

h_traffic_states = {'red_h', 'green_h', 'yellow_h', 'h_walk', 'both_red_h'}
h_trans_must = {}
h_trans_may = {}
starts_h = {'green_h'}
h_input_alph = {'red_v', 'green_v', 'yellow_v'}
h_output_alph = {'red_h', 'green_h', 'yellow_h', 'h_walk'}

h_trans_may[('green_h', 'red_h')] = {('True', 'red_h', '!')}
h_trans_must[('red_h', 'h_walk')] = {('h_timer == 2', 'red_h', '!')}
h_trans_must[('h_walk', 'h_walk')] = {('True', 'red_h', '!'), ('True', 'h_walk', '!')}
h_trans_must[('h_walk', 'both_red_h')] = {('True', 'red_v', '?')}
h_trans_must[('both_red_h', 'green_h')] = {('h_timer == 2', 'red_v', '?')}
h_trans_must[('green_h', 'yellow_h')] = {('True', 'red_v', '?')}
# h_trans_must[('green_h', 'green_h')] = {('True', 'green_h', '!')}
# h_trans_must[('yellow_h', 'yellow_h')] = {('True', 'yellow_h', '!')}
# h_trans_must[('yellow_h', 'fail')] = {('True', 'green_v', '?'), ('True', 'yellow_v', '?')}
# h_trans_must[('green_h', 'fail')] = {('True', 'green_v', '?'), ('True', 'yellow_v', '?')}
h_trans_must[('yellow_h', 'red_h')] = {('h_timer == 5', 'red_h', '!')}
h = construct_contract_automaton(state_set = h_traffic_states, starts = starts_h, musttrans = h_trans_must, maytrans = h_trans_may, input_alphabet = h_input_alph, output_alphabet = h_output_alph)
	
v_traffic_states = {'red_v', 'green_v', 'yellow_v', 'v_walk', 'both_red_v'}
v_trans_must = {}
v_trans_may = {}
v_input_alph = {'red_h', 'green_h', 'yellow_h'}
v_output_alph = {'red_v', 'green_v', 'yellow_v', 'v_walk'}
starts_v = {'v_walk'}

v_trans_may[('green_v', 'red_v')] = {('True', 'red_v', '!')}
v_trans_must[('red_v', 'v_walk')] = {('v_timer == 2', 'red_v', '!')}
v_trans_must[('v_walk', 'v_walk')] = {('True', 'red_v', '!'), ('True', 'v_walk', '!')}
v_trans_must[('v_walk', 'both_red_v')] = {('True', 'red_h', '?')}
v_trans_must[('both_red_v', 'green_v')] = {('v_timer == 2', 'red_h', '?')}
v_trans_must[('green_v', 'yellow_v')] = {('True', 'red_h', '?')}
v_trans_must[('yellow_v', 'red_v')] = {('v_timer == 5', 'red_v', '!')}
# v_trans_must[('yellow_v', 'fail')] = {('True', 'green_h', '?'), ('True', 'yellow_h', '?')}
# v_trans_must[('green_v', 'fail')] = {('True', 'green_h', '?'), ('True', 'yellow_h', '?')}
# h_trans_must[('green_v', 'green_v')] = {('True', 'green_v', '!')}
# h_trans_must[('yellow_v', 'yellow_v')] = {('True', 'yellow_v', '!')}

v = construct_contract_automaton(state_set = v_traffic_states, starts = starts_v, musttrans = v_trans_must, maytrans = v_trans_may, input_alphabet = v_input_alph, output_alphabet = v_output_alph)

h.convert_to_digraph().render('h', view = True)
v.convert_to_digraph().render('v', view = True)

# road_states = ['full', 'not full', 'aux', 'start_road', 'aux2']
# road_trans_may = {}
# road_trans_must = {}
# starts_road = {'start_road'}
# road_trans_may[('full', 'not full')] = ('g_exit', 'can_exit', '!')
# road_trans_must[('full', 'aux2')] = ('num_cars < max_cap', '', '')
# road_trans_must[('aux2', 'not full')] = ('True', '', '')
# road_trans_may[('full', 'aux2')] = ('num_cars < max_cap', '', '')
# road_trans_may[('aux2', 'not full')] = ('True', '', '')

# road_trans_may[('start_road', 'full')] = ('num_cars == max_cap', '', '')
# road_trans_may[('start_road', 'not full')] = ('num_cars < max_cap', '', '')
# road_trans_must[('not full', 'aux')] = ('g_enter', 'can_enter', '!')
# road_trans_must[('aux', 'full')] = ('num_cars == max_cap', '', '')
# road_trans_may[('not full', 'aux')] = ('g_enter', 'can_enter', '!')
# road_trans_may[('aux', 'full')] = ('num_cars == max_cap', '', '')
# road_trans_must[('aux', 'not full')] = ('num_cars < max_cap', '', '')
# road_trans_may[('not full', 'not full')] = ('g_exit', 'can_exit', '!')

# road_auto = construct_contract_automaton(state_set = road_states, starts = starts_road, musttrans = road_trans_must, maytrans = road_trans_may)
# road_auto.convert_to_digraph().render('road', view = True)

composed = compose_contract(h, v)
composed.convert_to_digraph().render('traffic light', view = True)

