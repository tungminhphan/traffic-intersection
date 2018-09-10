#!/usr/local/bin/python
# coding: utf-8
# Automaton Class
# Steve Guo, Tung M. Phan
# California Institute of Technology
# July 12, 2018

from random import sample
import numpy as np
import math, itertools
from graphviz import Digraph

# General state class.
class State:
    def __init__(self, name, composite_list = None):
        if isinstance(name, str) or isinstance(name, int):
            TypeError('state name must be of type string or int!')
        if isinstance(name, int):
            name = str(name) # convert name to string
        self.name = name.upper() # convert name to uppercase
        if composite_list == None:
            self.composite_list = list()
            self.composite_list.append(self)

        else:
            self.composite_list = composite_list

def product(state1, state2):
    composite_list = state1.composite_list + state2.composite_list
    new_name = '('
    for state in composite_list:
        new_name += state.name + ', '
    new_name = new_name[:-2]
    new_name += ')'
    return State(new_name, composite_list)

#test case
#s1 = State(1)
#s2 = State(2)
#s3 = State(3)
#z = product(s3,product(s1,s2))
#print(z.name)

# General transition class. Transition is a string from the alphabet.
class Transition:
    def __init__(self, start = None, end = None, transition = None):
        self.startState = start
        self.endState = end
        self.transition = transition

    def set_start_state(self, start):
        self.startState = start

    def set_end_state(self, end):
        self.endState = end

    def set_transition(self, transition):
        self.transition = transition

    def show(self):
        return self.transition

# General transition class for guard (where the guard is a set of inequalities.)
class guardTransition(Transition):
    def __init__(self, start = None, end = None, guard = True, action = '', actionType = ''):
        Transition.__init__(self, start, end)
        self.guard = guard # guard should be a dictionary of inequalities
        # actually, now guard is a string representing a boolean
        self.action = action

        # actionType is ?, !, or #, corresponding to input, output, and internal respectively
        self.actionType = actionType


    def show(self):
        transtext = ''
        # guard can be string
        if isinstance(self.guard, str):
            transtext = self.guard

        elif self.guard == False:
            return transtext

        elif self.guard == True:
            transtext += 'True'
            transtext = transtext[:-2] # deletes last comma and space

        transtext += ' / '

        transtext += actionType + action

        return transtext

# General finite automaton. 
class Automaton:
    def __init__(self):
        self.alphabet = set()
        self.transitions_dict = {} # transitions_dict[state] is the set of transitions from that state
        self.startStates = None
        self.endStates = set()
        self.states = set()

    def add_state(self, state, end_state = 0, start_state = 0):
        if end_state:
            self.endStates.add(state)
        if start_state:
            self.startStates.add(state)

        self.states.add(state)
        self.transitions_dict[state] = set()

    def add_transition(self, transition):
        self.transitions_dict[transition.startState].add(transition)

    def simulate(self, is_printing = False):
        state = self.startState
        while state not in self.endStates:
            if is_printing:
                print(state + "  taking transitions, output, input")
            else:
                print("taking transitions, output, input")
            state, = sample(self.transitions_dict[state], 1).endState # sample from set of all transistions uniformly at random
        print("simulation has terminated or deadlocked!")

    def convert_to_digraph(self):
        automata = Digraph(comment = 'insert description parameter later?')

        for state in self.states:
            # adds nodes
            automata.attr('node', shape = 'circle', color='green', style='filled', fixedsize='false')
            if state in self.endStates:
                automata.attr('node', shape = 'doublecircle', fixedsize = 'false')
            if state in self.startStates:
                automata.attr('node', color = 'yellow', fixedsize = 'false')
            automata.node(state.text, state.text)

        # adds transitions
        for state in self.states:
            transit = self.transitions_dict[state] # this is a set of transitions from the state
            for trans in transit:
                if trans != False:
                    state2 = trans.endState
                    transtext = trans.show()
                    automata.edge(state.text, state2.text, label = transtext)

        return automata


class InterfaceAutomaton(Automaton):
    def __init__(self, inputAlphabet = set(), outputAlphabet = set(), internalAlphabet = set()):
        Automaton.__init__(self)
        self.input_alphabet = inputAlphabet
        self.output_alphabet = outputAlphabet
        self.internal_alphabet = internalAlphabet
        self.alphabet = self.input_alphabet.union(self.output_alphabet).union(self.internal_alphabet)

        # takes two guard transitions and returns their composition
    def compose_guard_trans(self, tr1, tr2):
        if tr1.action != tr2.action and '' not in [tr1.actionType, tr2.actionType]:
            return False

        if tr1.guard == True:
            guard = tr2.guard
        elif tr2.guard == True:
            guard = tr1.guard
        elif isinstance(tr1.guard, str) and isinstance(tr2.guard, str):
            guard = tr1.guard + ' ∧ ' + tr2.guard

        # assumption is that either both are inequalities or strings

        newStart = product(tr1.startState, tr2.startState)
        newEnd = product(tr1.endState, tr2.endState)

        if tr1.actionType == '':
            newType = tr2.actionType
            action = tr2.action
        elif tr2.actionType == '':
            newType = tr1.actionType
            action = tr1.action
        else:
            action = tr1.action
            if tr1.actionType == '?':
                if tr2.actionType == '?':
                    newType = '?'
                elif tr2.actionType in {'!', '#'}:
                    newType = '#'

            elif tr1.actionType == '!':
                if tr2.actionType == '!':
                    newType = '!'
                elif tr2.actionType in {'?', '#'}:
                    newType = '#'

            elif tr1.actionType == '#':
                newType = '#'

        return guardTransition(newStart, newEnd, guard, action, newType)

    # in the final composition, delete all transitions that are still waiting for input, since we can't take them
    # also removes all states without transitions
    def trim(self):
        for key in self.transitions_dict:

            # removes transitions
            to_remove = set()
            for trans in self.transitions_dict[key]:
                if trans == False or len(trans.inputs) > 0:
                    to_remove.add(trans)

            self.transitions_dict[key] = self.transitions_dict[key] - to_remove

        # removes states without transitions
        for key in self.transitions_dict:
            to_remove = set()
            if len(self.transitions_dict[key]) == 0:
                to_remove.add(key)
                self.states.remove(key)

        for key in to_remove:
            self.transitions_dict.pop(key, None)


# # Add this later to make testing easier
# need to change
def construct_automaton(statelist, translist, start, ends):
    new_interface = InterfaceAutomaton()
    stringstatedict = {}
    # statelist is a list of strings representing state names
    for state in statelist:
        newstate = State(state)
        new_interface.add_state(newstate)
        stringstatedict[state] = newstate

    new_interface.set_start_state(stringstatedict[start])

    for end in ends:
        new_interface.endStates.add(stringstatedict[end])
    # translist is a dictionary; the key is a tuple of strings representing the states of the transition, and the value is a tuple:
    # (guardtext, inputs, outputs, internal actions)
    # inputs, outputs, internal action are sets of strings
    for key in translist:
        state1 = stringstatedict[key[0]]
        state2 = stringstatedict[key[1]]

        words = translist[key][0].split()
        inp = translist[key][1]
        out = translist[key][2]
        inter = translist[key][3]

        if words[0] == 'True':
            guard = True
        else:
            guard = translist[key][0]

        new_interface.add_transition(guardTransition(state1, state2, guard, inp, out, inter))

    return new_interface
