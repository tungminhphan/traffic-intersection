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

# test case for state class

#s1 = State(1)
#s2 = State(2)
#s3 = State(3)
#z = product(s3,product(s1,s2))
#print(z.name)

# General transition class. Transition is a string from the alphabet.
class Transition:
    def __init__(self, start = None, end = None, label = None):
        self.startState = start
        self.endState = end
        self.label = label

    def set_start_state(self, start):
        self.startState = start

    def set_end_state(self, end):
        self.endState = end

    def set_label(self, label):
        self.label = label

    def get_start(self):
        return '(' + self.startState.name + ')'

    def get_end(self):
        return '(' + self.endState.name + ')'

    def get_label(self):
        return self.label

    def print_transition(self):
        print(self.get_start() + ' ----> ' + self.get_end())

    def show(self):
        return self.label

# test case for general transition class
#s1 = State(1)
#s2 = State(2)
#t = Transition(s1,s2, 'a')
#t.print_transition()
#print(t.show())

# General transition class for guard (where the guard is a set of inequalities.)
class guardTransition(Transition):
    def __init__(self, start = None, end = None, guard = True, action = None, actionType = None):
        Transition.__init__(self, start, end)
        self.guard = guard # guard should be a dictionary of inequalities
        # actually, now guard is a string representing a boolean

        # actionType is ?, !, or #, corresponding to input, output, and internal respectively
        if actionType not in {'?', '!', '#', ''}:
            TypeError('actionType must be either ?, !, or #!')
        self.actionType = actionType

        if self.actionType == '':
            self.action = ''
        elif isinstance(action, str):
            TypeError('action must be of type str!')
        self.action = action

    def show(self):
        transtext = ''
        # guard can be string
        if isinstance(self.guard, str):
            transtext = '[' + self.guard
        elif self.guard == False:
            return transtext
        elif self.guard == True:
            transtext += 'True'
        transtext += ' | ' + self.actionType + self.action + ']'
        return transtext

    def print_transition(self):
        print(self.get_start() + ' --[' + self.show() + ']--> ' + self.get_end())

# test case for the guard transition class
# s1 = State(1)
# s2 = State(2)
# t = guardTransition(start = s1, end = s2, guard = 'x > 3', label = 'test', action = 'a', actionType = '?')
# t.print_transition()

# General finite automaton. 
class Automaton:
    def __init__(self):
        self.alphabet = set()
        self.transitions_dict = {} # transitions_dict[state] is the set of transitions from that state
        self.startStates = set()
        self.endStates = set()
        self.states = set()

    def add_state(self, state, end_state = False, start_state = False):
        if end_state:
            self.endStates.add(state)
        if start_state:
            self.startStates.add(state)

        self.states.add(state)
        self.transitions_dict[state] = set()

    def add_transition(self, transition):
        self.transitions_dict[transition.startState].add(transition)

    def convert_to_digraph(self):
        automata = Digraph(comment = 'insert description parameter later?')

        for state in self.states:
            # adds nodes
            automata.attr('node', shape = 'circle', style='filled', fixedsize='true')
            if state in self.startStates:
                automata.attr('node', color = 'gray', shape = 'doublecircle', fixedsize = 'true')
            automata.node(state.name, state.name)

        # adds transitions
        for state in self.states:
            transit = self.transitions_dict[state] # this is a set of transitions from the state
            for trans in transit:
                if trans != False:
                    state2 = trans.endState
                    transtext = trans.show()
                    automata.edge(state.name, state2.name, label = transtext)
        return automata


class InterfaceAutomaton(Automaton):
    def __init__(self, inputAlphabet = set(), outputAlphabet = set(), internalAlphabet = set(), failStates = set()):
        Automaton.__init__(self)
        self.input_alphabet = inputAlphabet
        self.output_alphabet = outputAlphabet
        self.internal_alphabet = internalAlphabet
        self.alphabet = self.input_alphabet.union(self.output_alphabet).union(self.internal_alphabet)
        self.failStates = failStates

        # takes two guard transitions and returns their composition
    def compose_guard_trans(tr1, tr2):
        if tr1.action != tr2.action and '' not in [tr1.actionType, tr2.actionType] or tr1.guard == False or tr2.guard == False:
            return False
        if tr1.guard == True:
            guard = tr2.guard
        elif tr2.guard == True:
            guard = tr1.guard
        elif isinstance(tr1.guard, str) and isinstance(tr2.guard, str):
            guard = tr1.guard + ' ∧ ' + tr2.guard

        newStart = product(tr1.startState, tr2.startState)

        # assumption is that either both are inequalities or strings
        if tr1.actionType == '':
            newType = tr2.actionType
            action = tr2.action
        else:
            action = tr1.action
            newType = tr1.actionType
            if {tr1.actionType, tr2.actionType} == {'!', '?'} or tr2.actionType == '#':
                newType = '#'
        return guardTransition(newStart, newEnd, guard, action, newType)

    # in the final composition, delete all transitions that are still waiting for input, since we can't take them
    # also removes all states without transitions
    def trim(self):
        for key in self.transitions_dict:
            # removes transitions
            to_remove = set()
            for trans in self.transitions_dict[key]:
                if trans == False:
                    to_remove.add(trans)
            self.transitions_dict[key] = self.transitions_dict[key] - to_remove

        # removes states without transitions
        for key in self.transitions_dict:
            keys_to_remove = set()
            if len(self.transitions_dict[key]) == 0:
                to_remove.add(key)
                self.states.remove(key)

        for key in keys_to_remove:
            self.transitions_dict.pop(key, None)

def construct_automaton(statelist, translist, starts):
    new_interface = InterfaceAutomaton()
    string_state_dict = {}
    # statelist is a list of strings representing state names
    for state in statelist:
        newstate = State(state)
        new_interface.add_state(newstate)
        string_state_dict[state] = newstate
    for start in starts:
        new_interface.startStates.add(string_state_dict[start])
    # translist is a dictionary; the key is a tuple of strings representing the states of the transition, and the value is a list:
    # (guardtext, action, action_type)
    for key in translist:
        state1 = string_state_dict[key[0]]
        state2 = string_state_dict[key[1]]
        guard = translist[key][0]
        action = translist[key][1]
        action_type = translist[key][2]
        new_interface.add_transition(guardTransition(state1, state2, guard, action, action_type))
    return new_interface

# testing
statelist = {'0', '1', '2', '3'}
starts = {'0', '1'}
translist = {('0', '1'): ('x > 5', 'a', '?')}
translist[('1', '2')] = ('True', 'c', '!')
translist[('2', '0')] = ('True', 'a', '!')
translist[('3', '0')] = ('y >= 3', 'b', '!')
translist[('1', '0')] = ('z >= 3', 'b', '#')
construct_automaton(statelist, translist, starts).convert_to_digraph().render('test', view = True)
