#!/usr/local/bin/python
# coding: utf-8
# Automaton Class
# Steve Guo, Tung M. Phan
# California Institute of Technology
# July 12, 2018
from automaton import *
class ContractAutomaton(InterfaceAutomaton):
    def __init__(self, must = {}, may = {}):
        InterfaceAutomaton.__init__(self)
        self.must = must
        self.may = may
        self.transitions_dict = may
        # may and must are transition dictionaries

    def check_validity(self):
        # checks every must transition is a may transition
        for key in self.must:
            musttrans = self.must[key]
            maytrans = self.may[key]
            for must in musttrans:
                check = False
                for may in maytrans:
                    if check_simulation(must, may):
                        check = True

                if not check:
                    return False
        return True

    def remove_state(self, state):
        # deletes all transitions leading to that state
        for key in self.transitions_dict:
            transitions = transitions_dict[key]
            for trans in transitions:
                if trans.endState == state:
                    transitions_dict[key].remove(trans)
                    may[key].remove(trans)
                    if key in must:
                        must[key].remove(trans)

        # deletes the state
        self.transitions_dict.pop(state)
        self.may.pop(state)
        self.must.pop(state)
        self.states.remove(state)
        self.startStates.remove(state)


    def add_transition(self, transition, must = 0):
        self.may[transition.startState].add(transition)
        if must:
            self.must[transition.startState].add(transition)

    def get_must_interface(self):
        return InterfaceAutomaton(self.alphabet, self.must, self.startStates, self.endStates, self.failStates,
                self.states, self.input_alphabet, self.output_alphabet, self.internal_alphabet)

    def get_may_interface(self):
        return InterfaceAutomaton(self.alphabet, self.may, self.startStates, self.endStates, self.failStates, self.states, self.input_alphabet, self.output_alphabet, self.internal_alphabet)
    def set_interface_automaton(self, interface):
            self.alphabet = interface.alphabet
            self.input_alphabet = interface.input_alphabet
            self.output_alphabet = interface.output_alphabet
            self.internal_alphabet = interface.internal_alphabet
            self.transitions_dict = interface.transitions_dict
            self.states = interface.states
            self.startStates = interface.startStates
            self.endStates = interface.endStates
            self.fail_state = interface.fail_state

    def convert_to_digraph(self):
        automata = Digraph(format = 'pdf')

        for state in self.states.union({self.fail_state}):
            # adds nodes
            automata.attr('node', color = 'gray', shape = 'circle', style='filled', fixedsize='false')
            if state in self.startStates:
                automata.attr('node', color = 'gray', style='filled',  fixedsize = 'false', shape='invhouse')
            automata.node(state.name, state.name)

        # adds transitions
        for state in self.states.union({self.fail_state}):
            if state in self.may:
                maytransit = self.may[state] # this is a set of may transitions from the state
                for trans in maytransit:
                    if trans is not False:
                        state2 = trans.endState
                        automata.edge(state.name, state2.name, label = trans.show(), style = 'dotted')

            if state in self.must:
                musttransit = self.must[state]
                for trans in musttransit:
                    if trans is not False:
                        state2 = trans.endState
                        automata.edge(state.name, state2.name, label = trans.show())

        return automata

    def prune_illegal_state(self):
        #remove any states such that must does not imply may
        finished = False
        while not finished:
            finished = True
            for key in self.must:
                musttrans = must[key]
                maytrans = may[key]
                for must in musttrans:
                    check = False
                    for may in maytrans:
                        if check_simulation(must, may):
                            check = True

                    if not check:
                        self.remove_state(key)
                        finished = False

    def weakAlphabetProjection(self, contract):
        # adds may self-loops
        alphabetDifference = contract.alphabet - self.alphabet
        for state in self.states:
            for letter in alphabetDifference:
                selfloop = guardTransition(state, state, 'True', letter, '#')
                self.add_transition(selfloop, 0)

    def strongAlphabetProjection(self, contract):
        # adds must self-loops
        alphabetDifference = contract.alphabet - self.alphabet
        for state in self.states:
            for letter in alphabetDifference:
                selfloop = guardTransition(state, state, 'True', letter, '#')
                self.add_transition(selfloop, 1)

def compose_contract(cr_1, cr_2):
    cr_1 = cr_1.strongAlphabetProjection(cr_2)
    cr_2 = cr_2.strongAlphabetProjection(cr_1)

    must1 = cr_1.get_must_interface()
    must2 = cr_2.get_must_interface()
    may1 = cr_1.get_may_interface()
    may2 = cr_2.get_may_interface()

    mustAuto = compose_interface(must1, must2)
    mayAuto = compose_interface(may1, may2)

    may1 = cr_1.may
    may2 = cr_2.may
    must1 = cr_1.must
    must2 = cr_2.must

    # pruning
    for key in mayAuto.states:
        notFound = False
        # needs to fix the following since composite list might have length > 2
        key1 = key.composite_list[0]
        key2 = key.composite_list[1]
        for may in may1[key1]:
            for must in must2[key2]:
                if may.action == must.action and may.actionType == must.actionType:
                    if is_satisfiable(may.guard + ' ∧ ' + must.guard):
                        notFound = True
            if notFound:
                mayAuto.remove_state(key)
                mustAuto.remove_state(key)

        for may in may2[key2]:
            for must in must1[key1]:
                if may.action == must.action and may.actionType == must.actionType:
                    if is_satisfiable(may.guard + ' ∧ ' + must.guard):
                        notFound = True
            if notFound:
                mayAuto.remove_state(key)
                mustAuto.remove_state(key)

    contract = ContractAutomaton(must = mustAuto.transitions_dict, may = mayAuto.transitions_dict)
    contract.set_interface_automaton(mayAuto)
    return contract

def check_simulation(trans1, trans2):
    # checks if trans1 <= trans2, ie they have the same action, action type, and g_1 => g_2
    # TODO
    if trans1.action != trans2.action or trans1.actionType != trans2.actionType:
        return False

    pass

def is_satisfiable(guard):
    # TODO
    pass 

# Makes contract automaton
def construct_contract_automaton(state_set, musttrans, maytrans, starts):
    new_contract = ContractAutomaton()
    string_state_dict = dict()
    string_state_dict['fail'] = new_contract.fail_state # add failure state manually
    # state_set is a list of strings representing state names
    for state in state_set:
        newstate = State(state)
        new_contract.add_state(newstate)
        string_state_dict[state] = newstate

    for start in starts:
        new_contract.startStates.add(string_state_dict[start])

    for key in musttrans:
        state1 = string_state_dict[key[0]]
        state2 = string_state_dict[key[1]]
        guard = musttrans[key][0]
        action = musttrans[key][1]
        action_type = musttrans[key][2]
        new_contract.add_transition(guardTransition(start = state1, end = state2,  guard = guard, action = action, actionType = action_type), must = 1)

    for key in maytrans:
        state1 = string_state_dict[key[0]]
        state2 = string_state_dict[key[1]]
        guard = maytrans[key][0]
        action = maytrans[key][1]
        action_type = maytrans[key][2]
        new_contract.add_transition(guardTransition(start = state1, end = state2,  guard = guard, action = action, actionType = action_type), must = 0)

    new_contract.trim()
    return new_contract


may = dict()
state_set = {'0', '1', '2', '3'}
starts = {'0', '1'}
may = {('0', '1'): ('x > 5', 'a', '?')}
may[('1', '2')] = ('True', 'c', '!')
may[('2', '0')] = ('True', 'a', '!')
D =  construct_contract_automaton(state_set=state_set, starts=starts, musttrans = {}, maytrans = may)
if D.check_validity():
    D.convert_to_digraph().render('D', view = True)




