from contract_automaton import *
h_traffic_states = {'0', '1', '2', '3', '4', ''}
h_trans_must = {}
h_trans_may = {}
starts_h = {'1'}
h_input_alph = {'r_v', 'g_v', 'y_v'}
h_output_alph = {'r_h', 'g_h', 'y_h', 'h_walk'}

h_trans_may[('1', '0')] = {('True', 'r_h', '!')}
h_trans_must[('0', '3')] = {('h_timer == 2', 'r_h', '!')}
h_trans_must[('3', '3')] = {('True', 'r_h', '!'), ('True', 'h_walk', '!')}
h_trans_must[('3', '4')] = {('True', 'r_v', '?')}
h_trans_must[('4', '1')] = {('h_timer == 2', 'r_v', '?')}
h_trans_must[('1', '2')] = {('True', 'r_v', '?')}
h_trans_must[('2', '0')] = {('h_timer == 5', 'r_h', '!')}
h = construct_contract_automaton(state_set = h_traffic_states, starts = starts_h, musttrans = h_trans_must, maytrans = h_trans_may, input_alphabet = h_input_alph, output_alphabet = h_output_alph)

v_traffic_states = {'0', '1', '2', '3', '4'}
v_trans_must = {}
v_trans_may = {}
v_input_alph = {'r_h', 'g_h', 'y_h'}
v_output_alph = {'r_v', 'g_v', 'y_v', 'v_walk'}
starts_v = {'3'}

v_trans_may[('1', '0')] = {('True', 'r_v', '!')}
v_trans_must[('0', '3')] = {('v_timer == 2', 'r_v', '!')}
v_trans_must[('3', '3')] = {('True', 'r_v', '!'), ('True', 'v_walk', '!')}
v_trans_must[('3', '4')] = {('True', 'r_h', '?')}
v_trans_must[('4', '1')] = {('v_timer == 2', 'r_h', '?')}
v_trans_must[('1', '2')] = {('True', 'r_h', '?')}
v_trans_must[('2', '0')] = {('v_timer == 5', 'r_v', '!')}

v = construct_contract_automaton(state_set = v_traffic_states, starts = starts_v, musttrans = v_trans_must, maytrans = v_trans_may, input_alphabet = v_input_alph, output_alphabet = v_output_alph)

pedestrian_states = {'0'}
p_must = {}
p_may = {}
p_input_alph = {'v_walk', 'h_walk'}
p_may[('0', '0')] = {('can_walk', 'v_walk', '?'), ('can_walk', 'h_walk', '?')}
p = construct_contract_automaton(state_set = pedestrian_states, starts = pedestrian_states, musttrans = p_must, maytrans = p_may, input_alphabet = p_input_alph)
p.convert_to_digraph().render('pedestrian', view = True)

h.convert_to_digraph().render('horizontal', view = True)
v.convert_to_digraph().render('vertical', view = True)

lights = compose_contract(h, v)
lights.convert_to_digraph().render('lights', view = True)
lights_p = compose_contract(lights, p)
lights_p.convert_to_digraph().render('lights_pedestrian', view = True)
