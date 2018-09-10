#!/usr/local/bin/python
# coding: utf-8
# Automaton Class
# Steve Guo, Tung M. Phan
# California Institute of Technology
# July 12, 2018

from random import sample
import inequality as iq
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

        # this next part is obsolete
        else:
            for key in self.guard:
                ineq = self.guard[key]
                transtext += ineq.show() + ', '

            transtext = transtext[:-2] # deletes last comma and space

        transtext += ' / '

        transtext += actionType + action

        return transtext

# General finite automaton. 
class Automaton:
    def __init__(self):
        self.alphabet = set()
        self.transitions_dict = {} # transitions_dict[state] is the set of transitions from that state's TEXT
        self.startState = None
        self.endStates = set()
        self.states = set()

    def add_state(self, state, end_state = 0, start_state = 0):
        if end_state:
            self.endStates.add(state)
        if start_state:
            self.startState = state

        self.states.add(state)
        self.transitions_dict[state] = set()

    def add_transition(self, transition):
        self.transitions_dict[transition.startState].add(transition)

    def set_start_state(self, state):
        self.startState = state

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
            if state == self.startState:
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

class ComponentAutomaton(Automaton):
    def __init__(self, inputAlphabet = set(), outputAlphabet = set()):
        Automaton.__init__(self)
        self.input_alphabet = inputAlphabet
        self.output_alphabet = outputAlphabet
        self.alphabet = self.input_alphabet.union(self.output_alphabet)

        # takes two guard transitions and returns their composition
    def compose_guard_trans(self, tr1, tr2):

        if tr1.guard == True:
            guard = tr2.guard
        elif tr2.guard == True:
            guard = tr1.guard
        elif isinstance(tr1.guard, str) and isinstance(tr2.guard, str):
            guard = tr1.guard + ' ∧ ' + tr2.guard

        # assumption is that either both are inequalities or strings
        else:
            guard = iq.conjunct(tr1.guard, tr2.guard)

        newstart = product(tr1.startState, tr2.startState)
        newend = product(tr1.endState, tr2.endState)

        inter = (tr1.inputs.intersection(tr2.outputs)).union(
            tr1.outputs.intersection(tr2.inputs)).union(tr1.internals).union(tr2.internals)

        inp = tr1.inputs.union(tr2.inputs) - inter
        out = tr1.outputs.union(tr2.outputs) - inter
        # if len(inp.union(out).union(inter)) == 0:
        #     return False

        return guardTransition(newstart, newend, guard, inp, out, inter)

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

def compose_components(component_1, component_2):
    new_component = ComponentAutomaton()
    newalphabet = component_1.alphabet.union(component_2.alphabet)
    # figure out what to do with input/output alphabets? 
    dict1 = component_1.transitions_dict
    dict2 = component_2.transitions_dict
    for key1 in dict1:
        for key2 in dict2:
            newstate = product(key1, key2)

            new_component.add_state(newstate, key1 in component_1.endStates and key2 in component_2.endStates
                , key1 == component_1.startState and key2 == component_2.startState)

            for trans1 in dict1[key1]:
                for trans2 in dict2[key2]:
                    new_component.transitions_dict[newstate].add(new_component.compose_guard_trans(trans1, trans2))

    new_component.alphabet = newalphabet
    return new_component

def compose_multiple_components(list_components):
    curr_component = list_components[0]
    for comp in list_components[1:]:
        curr_component = compose_components(curr_component, comp)

    return curr_component

class ContractAutomaton(Automaton):
    def __init__(self, must = {}, may = {}):
        Automaton.__init__(self)
        self.must = must
        self.may = may
        # may and must are transition dictionaries

    def check_validity(self):
        # checks that the states are equal
        for key in may.states:
            for key2 in must.states:
                if key not in must.states:
                    return False
                if key2 not in may.states:
                    return False

        # checks every may transition is a must transition
        for key in may.transitions_dict:
            trans = may.transitions_dict[key]
            for transition in trans:
                if transitions not in must.transitions_dict[key]:
                    return False

        return True

        def add_transition(self, transition, must = 0):
            self.may[transition.startState].add(transition)
            if must:
                self.must[transition.startState].add(transition)

        def convert_to_digraph(self):
            automata = Digraph(comment = 'insert description parameter later?')
            maxlen = 0
            for state in self.states:
                if len(state.text) > maxlen:
                    maxlen = len(state.text)

            for state in self.states:
                # adds nodes
                automata.attr('node', shape = 'circle', color='yellow', style='filled', fixedsize='true')
                if state in self.endStates:
                    automata.attr('node', shape = 'doublecircle', fixedsize = 'true')
                if state == self.startState:
                    automata.attr('node', color = 'yellow', fixedsize = 'true')
                newtext = ' ' * math.floor((maxlen - len(state.text))/2) + state.text + ' ' * math.ceil((maxlen - len(state.text))/2)
                automata.node(newtext, newtext)

            # adds transitions
            for state in self.states:
                maytransit = self.may[state] # this is a set of may transitions from the state
                musttransit = self.must[state]
                for trans in maytransit:
                    if trans != False:
                        state2 = trans.endState
                        transtext = trans.show()
                        automata.edge(state.text, state2.text, label = transtext, style = 'dotted')

                for trans in musttransit:
                    if trans != False:
                        state2 = trans.endState
                        transtext = trans.show()
                        automata.edge(state.text, state2.text, label = transtext)

            return automata

# assumes weight on a graph is a string of the form "guard / ?input, !output, #internal separated by , "
def convert_graph_to_automaton(digraph):
    new_component = ComponentAutomaton()
    nodes = digraph._nodes
    edges = digraph._edges
    stringstatedict = {}
    inp = set()
    out = set()
    inter = set()

    for node in nodes:
        newstate = State(node)
        new_component.add_state(node)
        stringstatedict[node] = newstate

    for source in sources:
        new_component.set_start_state(stringstatedict[source])

    for sink in sinks:
        new_component.endStates.add(stringstatedict[sink])


    for trans in edges:
        state1 = stringstatedict[trans[0]]
        state2 = stringstatedict[trans[1]]
        text = digraph._weights[trans]
        words = text.split() # weight is a string

        actions = text[text.find('/') + 1:].split() # part of transition text after / delimiter

        if words[0] == 'True':
            guard = True

        else:
            if len(words) > 3 and words[1] == '≤' and words[3] == '≤':
                lwrbnd = float(words[0])
                var = words[2]
                uprbnd = float(words[4])
            elif words[1] == '≥':
                var = words[0]
                lwrbnd = float(words[2])
                uprbnd = np.inf
            elif words[1] == '≤':
                var = words[0]
                uprbnd = float(words[2])
                lwrbnd = -np.inf

        guard = iq.dictionarize(iq.Inequality(var, lwrbnd, uprbnd))

        # Each action in form ?(input), !(output), or #(internal)
        for action in actions:
            if action[0] == '?':
                inp.add(action[1:])
            elif action[0] == '!':
                out.add(action[1:])
            elif action[0] == '#':
                inter.add(action[1:])
            else:
                raise SyntaxError('Input, output or internals in wrong format.')

        new_component.add_transition(guardTransition(state1, state2, guard, inp, out, inter))

    return new_component


# # Add this later to make testing easier
# currently only works if the guard text is a single inequality
def construct_automaton(statelist, translist, start, ends):
    new_component = ComponentAutomaton()
    stringstatedict = {}
    # statelist is a list of strings representing state names
    for state in statelist:
        newstate = State(state)
        new_component.add_state(newstate)
        stringstatedict[state] = newstate

    new_component.set_start_state(stringstatedict[start])

    for end in ends:
        new_component.endStates.add(stringstatedict[end])

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
        elif len(words) > 3 and words[1] == '≤' and words[3] == '≤':
            lwrbnd = float(words[0])
            var = words[2]
            uprbnd = float(words[4])
            guard = iq.dictionarize(iq.Inequality(var, lwrbnd, uprbnd))
        elif len(words) > 2 and words[1] == '≥':
            var = words[0]
            lwrbnd = float(words[2])
            uprbnd = np.inf
            guard = iq.dictionarize(iq.Inequality(var, lwrbnd, uprbnd))
        elif len(words) > 2 and words[1] == '≤':
            var = words[0]
            uprbnd = float(words[2])
            lwrbnd = -np.inf
            guard = iq.dictionarize(iq.Inequality(var, lwrbnd, uprbnd))
        else:
            guard = translist[key][0]

        new_component.add_transition(guardTransition(state1, state2, guard, inp, out, inter))

    return new_component
