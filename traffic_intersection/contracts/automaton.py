# Try 2!

from random import sample
import numpy as np
import math
import itertools
from graphviz import Digraph

# General state class. Has a 'set' property for the purposes of composition.
class State:
    def __init__(self, text, newset = None):
        self.text = text.upper()
        if newset == None:
            self.set = set()
            self.set.add(self)
        else:
            self.set = newset

def product(state1, state2):
    newset = state1.set.union(state2.set)
    newtext = '('
    for state in state1.set:
        newtext += state.text + ', '
    for state in state2.set:
        newtext += state.text + ', '
    newtext = newtext[:-2]
    newtext += ')'
    return State(newtext, newset)


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
    def __init__(self, alphabet = set(), transitions = {}, startStates = set(), endStates = set(), failStates = set(), states = set()):
        self.alphabet = alphabet
        self.transitions_dict = transitions # transitions_dict[state] is the set of transitions from that state
        self.startStates = startStates
        self.endStates = endStates
        self.failStates = failStates
        self.states = states

    def add_state(self, state, end_state = 0, start_state = 0, fail_state = 0):
        if end_state:
            self.endStates.add(state)
        if start_state:
            self.startStates.add(state)
        if fail_state:
            self.failStates.add(state)

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

        elif tr2.actionType = '':
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


def compose_interface(c_1, c_2):
    newInput = c_1.input_alphabet.union(c_2.inputAlphabet)
    newOutput = c_1.output_alphabet.union(c_2.output_alphabet)

    newInternal = c_1.internal_alphabet.union(c_2.internal_alphabet).union(newInput.intersection(newOutput))

    new_interface = InterfaceAutomaton(inputAlphabet = newInput, outputAlphabet = newOutput, internalAlphabet = newInternal)

    dict1 = c_1.transitions_dict
    dict2 = c_2.transitions_dict
    for key1 in dict1:
        for key2 in dict2:
            newstate = product(key1, key2)

            new_interface.add_state(newstate, key1 in c_1.endStates and key2 in c_2.endStates
                , key1 in c_1.startStates and key2 in c_2.startStates, key1 in c_1.failStates
                or key2 in c_2.failStates)

            for trans1 in dict1[key1]:
                for trans2 in dict2[key2]:
                    new_interface.transitions_dict[newstate].add(new_interface.compose_guard_trans(trans1, trans2))

    new_interface.alphabet = newalphabet
    return new_interface

def compose_multiple_interfaces(list_interfaces):
    curr_interface = list_interfaces[0]
    for comp in list_interfaces[1:]:
        curr_interface = compose_interface(curr_interface, comp)

    return curr_interface
