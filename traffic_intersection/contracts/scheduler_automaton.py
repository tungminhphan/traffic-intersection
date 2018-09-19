from contract_automaton import *

scheduler_states = {'0', '1', '2'}
trans_must = {}
trans_may = {}
starts_h = {'0'}
input_alph = {'request'}
output_alph = {'reject', 'accept', 'primitives'}
internal_alph = {'reject', 'accept', 'primitives'}

trans_may[('1', '0')] = {('True', 'reject', '!')}
trans_must[('2', '0')] = {('True', 'primitives', '!')}
trans_must[('1', '2')] = {('True', 'accept', '!')}
trans_must[('0', '0')] = {('True', 'request', '?')}
trans_must[('0', '1')] = {('len(request_queue)>0', 'processing', '#')}
scheduler = construct_contract_automaton(state_set = scheduler_states, starts = starts_h, musttrans = trans_must, maytrans = trans_may, input_alphabet = input_alph, output_alphabet = output_alph, internal_alphabet = internal_alph)


scheduler.convert_to_digraph().render('scheduler', view = True)

