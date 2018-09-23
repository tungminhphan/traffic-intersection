#!/usr/local/bin/python
# coding: utf-8
# Automaton Class
# Steve Guo, Tung M. Phan
# California Institute of Technology
# July 12, 2018

from automaton import *
class ContractAutomaton(InterfaceAutomaton):
    def __init__(self, must = {}, may = {}):
        self.must = must
        self.transitions_dict = may
        InterfaceAutomaton.__init__(self)
        # may and must are transition dictionaries

    def check_validity(self):
        # checks every must transition is a may transition
        for key in self.must:
            musttrans = self.must[key]
            maytrans = self.transitions_dict[key]
            for must in musttrans:
                check = False
                for may in maytrans:
                    if check_simulation(must, may):
                        check = True

                if not check:
                    return False
        return True

    def add_state(self, state, end_state = False, start_state = False):
        if end_state:
            self.endStates.add(state)
        if start_state:
            self.startStates.add(state)

        self.states.add(state)
        self.transitions_dict[state] = set()
        self.must[state] = set()

    def remove_state(self, state):
        # deletes all transitions leading to that state
        for key in self.transitions_dict:
            maytransitions = self.transitions_dict[key]
            to_remove_may = set()
            to_remove_must = set()

            for trans in maytransitions:
                if trans.endState == state:
                    to_remove_may.add(trans)

            for trans in to_remove_may:
                    self.transitions_dict[key].remove(trans)

            musttransitions = self.must[key]

            for trans in musttransitions:
                if trans.endState == state:
                    to_remove_must.add(trans)

            for trans in to_remove_must:
                self.must[key].remove(trans)

        # deletes the state
        self.transitions_dict.pop(state)
        self.must.pop(state)
        self.states.remove(state)
        try:
            self.startStates.remove(state)
        except KeyError:
            pass
    def add_transition(self, transition, must = 0, may = 1):
        if transition != False:
            self.alphabet.add(transition.action)
            if transition.actionType == '?':
                self.input_alphabet.add(transition.action)
            elif transition.actionType == '!':
                self.output_alphabet.add(transition.action)
            elif transition.actionType =='#':
                self.internal_alphabet.add(transition.action)

            if may:
                self.transitions_dict[transition.startState].add(transition)

            if must:
                self.must[transition.startState].add(transition)

    def add_implicit_self_transitions(self):
        for state in self.transitions_dict:
            transition = guardTransition(state, state, 'True', 'eps', '')
            self.add_transition(transition, 0)

    def get_must_interface(self):
        interface = InterfaceAutomaton(self.input_alphabet, self.output_alphabet, self.internal_alphabet, self.fail_state)
        interface.states = self.states
        interface.startStates = self.startStates
        interface.endStaets = self.endStates
        interface.transitions_dict = self.must
        return interface

    def get_may_interface(self):
        interface = InterfaceAutomaton(self.input_alphabet, self.output_alphabet, self.internal_alphabet, self.fail_state)
        interface.states = self.states
        interface.startStates = self.startStates
        interface.endStates = self.endStates
        interface.transitions_dict = self.transitions_dict
        return interface

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
        automata = Digraph(format = 'svg')
        for state in self.states.union({self.prestart_state}):
            # adds nodes
            automata.attr('node', color = 'gray', shape = 'circle', style='filled', fixedsize='false')
            if state is self.prestart_state:
                automata.attr('node', color = 'white',  fixedsize = 'false', shape='point')
            automata.node(state.name, state.name)
        # adds transitions
        for state in self.startStates:
            automata.edge(self.prestart_state.name, state.name, label = 'start')

        for state in self.states.union({self.fail_state}):
            if state in self.transitions_dict:
                maytransit = self.transitions_dict[state] # this is a set of may transitions from the state
                for trans in maytransit:
                    if trans is not False:
                        checked = True
                        state2 = trans.endState
                        for musttrans in self.must[state]:
                            if musttrans.endState == state2 and musttrans.actionType == trans.actionType and musttrans.action == trans.action:
                                checked = False
                                trans2 = musttrans
                                if trans2.show() == trans.show() or trans2.guard == 'True':
                                    automata.edge(state.name, state2.name, label = trans2.show())
                                else:
                                    automata.edge(state.name, state2.name,
                                            label = '[' + trans.guard + '∧¬'
                                            + trans2.guard + ' | ' +
                                            trans.actionType + trans.action +
                                            ']')
                                    automata.edge(state.name, state2.name, label = '['  + trans2.guard + ' | ' + trans.actionType + trans.action + ']')
                                if not checked:
                                    break
                        if checked:
                                automata.edge(state.name, state2.name, label = trans.show(), style = 'dashed')
        return automata

    def prune_illegal_state(self):
        #remove any states such that must does not imply may
        finished = False
        while not finished:
            finished = True
            for key in self.states:
                musttrans = self.must[key]
                maytrans = self.transitions_dict[key]
                print(key.name)
                for must in musttrans:
                    check = False
                    for may in maytrans:
                        if check_simulation(must, may):
                            check = True
                    if not check:
                        self.remove_state(key)
                        finished = False

    def alphabetProjection(self, contract, strong = 0):
        # if strong, adds must self-loop
        input_alphabet_difference = set()
        output_alphabet_difference = set()
        internal_alphabet_difference = set()
        alphabet_difference = contract.alphabet - self.alphabet
        for letter in alphabet_difference:
            if letter in contract.input_alphabet:
                input_alphabet_difference.add(letter)
            if letter in contract.output_alphabet:
                output_alphabet_difference.add(letter)
            if letter in contract.internal_alphabet:
                internal_alphabet_difference.add(letter)

        for state in self.states - {self.fail_state}:
            for letter in input_alphabet_difference:
                selfloop = guardTransition(state, state, 'True', letter, '?')
                self.add_transition(selfloop, strong)

            for letter in output_alphabet_difference:
                selfloop = guardTransition(state, state, 'True', letter, '!')
                self.add_transition(selfloop, strong)

            for letter in internal_alphabet_difference:
                selfloop = guardTransition(state, state, 'True', letter, '#')
                self.add_transition(selfloop, strong)


def compose_contract(cr_1, cr_2):
    cr_1.alphabetProjection(cr_2, 1)
    cr_2.alphabetProjection(cr_1, 1)
    new_contract = ContractAutomaton()

    node_dict = dict() # maintain references to states being composed
    for key1 in cr_1.states:
        for key2 in cr_2.states:
            newstate = product(key1, key2)
            if key1 == cr_1.fail_state or key2 == cr_2.fail_state:
                node_dict[(key1, key2)] = new_contract.fail_state
            else:
                node_dict[(key1, key2)] = product(key1, key2)

    for key1 in cr_1.states:
        for key2 in cr_2.states:
            newstate = node_dict[(key1, key2)]
            new_contract.add_state(newstate, start_state = key1 in cr_1.startStates and key2 in cr_2.startStates)

            for trans1 in cr_1.transitions_dict[key1]:
                for trans2 in cr_2.transitions_dict[key2]:
                    new_trans = compose_guard_trans(trans1, trans2, node_dict)
                    if new_trans != False:
                        new_contract.transitions_dict[newstate].add(new_trans)
                        new_contract.alphabet.add(new_trans.action)
                        if new_trans.actionType == '?':
                            new_contract.input_alphabet.add(new_trans.action)
                        elif new_trans.actionType == '!':
                            new_contract.output_alphabet.add(new_trans.action)
                        elif new_trans.actionType =='#':
                            new_contract.internal_alphabet.add(new_trans.action)

            for trans1 in cr_1.must[key1]:
                for trans2 in cr_2.must[key2]:
                    if compose_guard_trans(trans1, trans2, node_dict) != False:
                        new_contract.must[newstate].add(compose_guard_trans(trans1, trans2, node_dict))

    new_contract.trim()
    return new_contract

def conjunct_contract(cr_1, cr_2):
    cr_1.alphabetProjection(cr_2, 0)
    cr_2.alphabetProjection(cr_1, 0)
    new_contract = ContractAutomaton()
    node_dict = dict() # maintain references to states being composed
    for key1 in cr_1.states:
        for key2 in cr_2.states:
            newstate = product(key1, key2)
            if key1 == cr_1.fail_state or key2 == cr_2.fail_state:
                node_dict[(key1, key2)] = new_contract.fail_state
            else:
                node_dict[(key1, key2)] = product(key1, key2)

    transdict = {} # key is (state, action, actiontype), value is a transition (if exists) from that state with that action/actiontype

    # need to add states separately first
    for key1 in cr_1.states:
        for key2 in cr_2.states:
            newstate = node_dict[(key1, key2)]
            new_contract.add_state(newstate, start_state = key1 in cr_1.startStates and key2 in cr_2.startStates)

    for key1 in cr_1.states:
        for key2 in cr_2.states:

            for trans1 in cr_1.transitions_dict[key1]:
                for trans2 in cr_2.transitions_dict[key2]:
                    new_trans = conjunct_may_trans(trans1, trans2, node_dict)
                    if new_trans != False:
                        new_contract.add_transition(new_trans)

            # adds correct transition if both key1 and key2 have the same action type transition from them
            for trans1 in cr_1.must[key1]:
                for trans2 in cr_2.must[key2]:
                    new_trans = conjunct_must_trans(trans1, trans2, node_dict)
                    if new_trans != False:
                        new_contract.add_transition(new_trans, 1, 0)
                        transdict[(newstate, new_trans.action, new_trans.actionType)] = new_trans


            for trans1 in cr_1.must[key1]:
                # the following implies the transition w/ this actiontype is only in the first contract from this state, and the transition remains the same
                if (newstate, trans1.action, trans1.actionType) not in transdict:
                    startState = node_dict[(trans1.startState, trans2.startState)]
                    endState = node_dict[(trans1.endState, trans2.endState)]
                    new_trans = guardTransition(startState, endState, trans1.guard, trans1.action, trans1.actionType)
                    new_contract.add_transition(new_trans, 1, 0)

            for trans2 in cr_2.must[key2]:
                # the following implies the transition w/ this actiontype is only in the first contract from this state, and the transition remains the same
                if (newstate, trans2.action, trans2.actionType) not in transdict:
                    startState = node_dict[(trans1.startState, trans2.startState)]
                    endState = node_dict[(trans1.endState, trans2.endState)]
                    new_trans = guardTransition(startState, endState, trans2.guard, trans2.action, trans2.actionType)
                    new_contract.add_transition(new_trans, 1, 0)

    new_contract.trim()
    new_contract.prune_illegal_state()
    return new_contract



def check_simulation(trans1, trans2):
    # checks if trans1 <= trans2, ie they have the same action, action type, and g_1 => g_2
    # TODO
    if trans1.action != trans2.action or trans1.actionType != trans2.actionType:
        return False

    return True

def is_satisfiable(guard):
    # TODO
    pass

# Makes contract automaton
# change later so alphabet is better
def construct_contract_automaton(state_set, musttrans, maytrans, starts, input_alphabet = set(), output_alphabet = set(), internal_alphabet = set()):
    new_contract = ContractAutomaton()

    new_contract.input_alphabet = input_alphabet
    new_contract.output_alphabet = output_alphabet
    new_contract.internal_alphabet = internal_alphabet
    new_contract.alphabet = input_alphabet.union(output_alphabet).union(internal_alphabet)

    string_state_dict = dict()
    string_state_dict['⊥'] = new_contract.fail_state # add failure state manually
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
        for trans in musttrans[key]:
            guard = trans[0]
            action = trans[1]
            action_type = trans[2]
            new_contract.add_transition(guardTransition(start = state1, end = state2,  guard = guard, action = action, actionType = action_type), must = 1)

    for key in maytrans:
        state1 = string_state_dict[key[0]]
        state2 = string_state_dict[key[1]]
        for trans in maytrans[key]:
            guard = trans[0]
            action = trans[1]
            action_type = trans[2]
            new_contract.add_transition(guardTransition(start = state1, end = state2,  guard = guard, action = action, actionType = action_type), must = 0)
    new_contract.trim()
    return new_contract


