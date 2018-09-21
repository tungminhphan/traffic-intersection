from contract_automaton import *
h_traffic_states = {'0', '1', '2', '3', '4'}
h_trans_must = {}
h_trans_may = {}
starts_h = {'0'}
h_input_alph = {'r_v', 'g_v', 'y_v'}
h_output_alph = {'r_h', 'g_h', 'y_h', 'h_walk'}

h_trans_may[('0', '1')] = {('(h=g)∧(h\'=r)','r_h','!')}
h_trans_must[('0', '2')] = {('(h=g)∧(h\'=y)', 'r_v', '?')}
h_trans_must[('2', '1')] = {('(h=y)∧(h\'=r)\n∧(h_timer>=t_y)\n∧(h_timer\'= 0)', 'r_h', '!')}
h_trans_must[('1', '3')] = {('(h=r)∧(h\'=r)\n∧(h_timer>=t_c)\n∧(h_timer\'=0)', 'r_h', '!')}
h_trans_may[('1', '4')] = {('(h=r)∧(h\'=r)', 'r_v', '?')}
h_trans_must[('3', '3')] = {('(h=r)∧(h\'=r)\n∧(h_timer<t_w)', 'h_walk', '!'), ('(h=r)∧(h\'=r)', 'r_h', '!')}
h_trans_must[('3', '4')] = {('(h=r)∧(h\'=r)\n∧(h_timer>=t_w)\n∧(h_timer\'=0)', 'r_v', '?')}
h_trans_must[('4', '0')] = {('(h=r)∧(h\'=g)', 'r_v', '?')}
#h_trans_must[('3', '⊥')] = {('¬r_h', 'ε', '')}
h = construct_contract_automaton(state_set = h_traffic_states, starts = starts_h, musttrans = h_trans_must, maytrans = h_trans_may, input_alphabet = h_input_alph, output_alphabet = h_output_alph)

v_traffic_states = {'0', '1', '2', '3', '4'}
v_trans_must = {}
v_trans_may = {}
starts_v = {'3'}
v_input_alph = {'r_h', 'g_h', 'y_h'}
v_output_alph = {'r_v', 'g_v', 'y_v', 'v_walk'}
v_trans_may[('0', '1')] = {('(v=g)∧(v\'=r)', 'r_v', '!')}
v_trans_must[('0', '2')] = {('(v=g)∧(v\'=y)', 'r_h', '?')}
v_trans_must[('2', '1')] = {('(v=y)∧(v\'=r)\n∧(v_timer>=t_y)\n∧(v_timer\'=0)', 'r_v', '!')}
v_trans_must[('1', '3')] = {('(v=r)∧(v\'=r)\n∧(v_timer>=t_c)\n∧(v_timer\'=0)', 'r_v', '!')}
v_trans_may[('1', '4')] = {('(v=r)∧(v\'=g)\n', 'r_h', '?')}
v_trans_must[('3', '3')] = {('(v=r)∧(v\'=r)\n∧(v_timer<t_w)', 'v_walk', '!'), ('(v=r)∧(v\'=r)', 'r_v', '!')}
v_trans_must[('3', '4')] = {('(v=r)∧(v\'=r)\n∧(v_timer>=t_w)\n∧(v_timer\'=0)', 'r_h', '?')}
v_trans_must[('4', '0')] = {('(v=r)∧(v\'=g)', 'r_h', '?')}

v = construct_contract_automaton(state_set = v_traffic_states, starts = starts_v, musttrans = v_trans_must, maytrans = v_trans_may, input_alphabet = v_input_alph, output_alphabet = v_output_alph)

pedestrian_states = {'0'}
starts_p = {'0'}
p_must = {}
p_may = {}
p_input_alph = {'v_walk', 'h_walk'}
p_may[('0', '0')] = {('(t_cross <= t_w)', 'v_walk', '?'), ('(t_cross <= t_w)', 'h_walk', '?')}
p = construct_contract_automaton(state_set = pedestrian_states, starts = starts_p, musttrans = p_must, maytrans = p_may, input_alphabet = p_input_alph)
lights = compose_contract(h, v)
p.convert_to_digraph().render('pedestrian', view = True)
lights_p = compose_contract(lights, p)
h.convert_to_digraph().render('horizontal', view = False)
v.convert_to_digraph().render('vertical', view = False)
lights.convert_to_digraph().render('lights', view = False)
lights_p.convert_to_digraph().render('lights_pedestrian', view = True)
