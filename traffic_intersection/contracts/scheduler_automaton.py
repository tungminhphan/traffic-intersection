from contract_automaton import *

car_states = {'0', '1', '2', '3'}
car_must = {}
car_may = {}
car_start = {'0'}
car_input_alph = {'reject', 'accept', 'primitives'}
car_output_alph = {'request'}
car_must[('0', '1')] = {('True', 'request', '!')}
car_must[('1', '2')] = {('True', 'accept', '?')}
car_must[('1', '0')] = {('True', 'reject', '?')}
car_must[('2', '3')] = {('True', 'primitives', '?')}
car_may[('3', '3')] = {('True', 'primitives', '?')}
car_may[('3', '1')] = {('not_done', 'request', '!')}
car_auto = construct_contract_automaton(state_set = car_states, starts = car_start, musttrans = car_must, maytrans = car_may, input_alphabet = car_input_alph, output_alphabet = car_output_alph)
car_auto.convert_to_digraph().render('car', view = True)

scheduler_states = {'0', '1', '2', '3'}
trans_must = {}
trans_may = {}
starts_h = {'0'}
input_alph = {'request'}
output_alph = {'reject', 'accept', 'primitives'}
internal_alph = {'processing'}

trans_may[('2', '0')] = {('True', 'reject', '!')}
#trans_must[('1', '1')] = {('True', 'request', '?')}
trans_must[('3', '0')] = {('True', 'primitives', '!')}
trans_must[('2', '3')] = {('True', 'accept', '!')}
trans_must[('0', '1')] = {('True', 'request', '?')}
trans_must[('1', '2')] = {('len(request_queue)>0', 'processing', '#')}
scheduler = construct_contract_automaton(state_set = scheduler_states, starts = starts_h, musttrans = trans_must, maytrans = trans_may, input_alphabet = input_alph, output_alphabet = output_alph, internal_alphabet = internal_alph)
scheduler.convert_to_digraph().render('scheduler', view = True, dpi = 300)

sched_car = compose_contract(scheduler, car_auto)
sched_car.convert_to_digraph().render('sched_car', view = True)
