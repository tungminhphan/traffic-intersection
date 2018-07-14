# Contract Automaton Class
# Tung M. Phan
# California Institute of Technology
# May 8, 2018



from random import sample
import math
import itertools
from graphviz import Digraph

# Inequality for guard. Left-hand side is variable name, and is bounded on both sides (if one-sided inequality, has +/- infty on other side.)
class Inequality:
    def __init__(self, left = '', lower = -math.inf, upper = math.inf):
        self.lhs = left # a variable
        self.lwrbnd = lower # a number
        self.uprbnd = upper 

# returns an inequality that has the same truth value as the conjunction of two inequalities
def conjunct(ineq1, ineq2):
    if ineq1 == True:
        return ineq2
    elif ineq2 == True:
        return ineq1

    if ineq1.lhs == ineq2.lhs:
        return Inequality(ineq1.lhs, max(ineq1.lwrbnd, ineq2.lwrbnd), min(ineq1.uprbnd, ineq2.uprbnd))
    else:
        return (ineq1, ineq2)

# returns an inequality that has the same truth value as disjunction of two inequalities
def disjunct(ineq1, ineq2):
    if ineq1.lhs == ineq2.lhs:
        return Inequality(guard1.lhs, min(ineq1.lwrbnd, ineq2.lwrbnd), max(ineq1.uprbnd, ineq2.uprbnd))
    else:
        return False # ? don't need to ever use this in this case

# returns a set of inequalities whose conjunction has the same truth value as the conjunction of two sets of inequalities
def conjuncset(set1, set2):
    set3 = []
    for ineq1 in set1:
        for ineq2 in set2:
            if ineq1.lhs == ineq2.lhs:
                set3.append(conjunct(ineq1, ineq2))
    return set3

# returns a set of inequalities that has the same truth value as the disjunction of two sets of inequalities
def disjunctset(set1, set2):
    return True

# Transition class. Has guard and optional input/output/internal action
class Transition:
    def __init__(self, s1 = '', s2 = '', ineq = True, inp = [], out = [], inter = []):
        self.guard = ineq # guard is a set of inequalities joined by conjunction (?)
        self.input = inp # set of inputs
        self.output = out
        self.internal = inter # internal action?
        self.state1 = s1 # set
        self.state2 = s2 # transition from s1 to s2

def compose_transitions(tr1, tr2):
    guard = conjunct(tr1.guard, tr2.guard)

    input1 = set(tr1.input)
    input2 = set(tr2.input)
    output1 = set(tr1.output)
    output2 = set(tr2.output)

    inp = input1.union(input2)
    inter = (input1.intersection(output2)).union(output1.intersection(input2))
    out = output1.intersection(output2)
    return Transition(guard, inp, out, inter, (tr1.state1, tr2.state1), (tr1.state2, tr2.state2))


class ComponentAutomaton:
    def __init__(self):
        self.alphabet = set() # the alphabet for the component
        self.transitions_dict = {} # dictionary of all transitions
        self.startState = None
        self.endStates = []
        self.states = []

    def add_state(self, name, transitions, end_state = 0):
        if end_state:
            self.endStates.append(name)
        # self.transitions_dict[name.upper()] = map(lambda x: x.upper(), transitions) # transitions is a set of transitions
        self.transitions_dict[name] = transitions
        self.states.append(name)

    def set_start_state(self, name):
        self.startState = name

    # doesn't work with current definition of transition
    def simulate(self, is_printing = False):
        state = self.startState
        while state not in self.endStates:
            if is_printing:
                print(state + "  taking transitions, output, input")
            else:
                print("taking transitions, output, input")
            state, = sample(self.transitions_dict[state],1) # sample from set of all transistions uniformly at random 
        print("simulation has terminated or deadlocked!")

    def convert_component(self):
        automata = Digraph(comment = 'insert description parameter later?')
        # i think this automata class is missing the possibility of states without transitions? not that they matter
        for key in self.states:
            # adds nodes
            automata.attr('node', shape = 'circle', color='green', style='filled', fixedsize='false', width='1')
            if key in self.endStates:
                automata.attr('node', shape = 'doublecircle')
            if key == self.startState:
                automata.attr('node', color = 'yellow')
            automata.node(str(key), str(key)) # i think this is the syntax?

        # adds transitions
        for key in self.states:
            transitions = self.transitions_dict[key]
            for trans in transitions:
                q = trans.state1 # assuming the states are strings?
                qprime = trans.state2
                inputs = trans.input
                outputs = trans.output
                internals = trans.internal
                text = ''

                guard = trans.guard
                if guard == True:
                    text += 'T'
                else:
                    lwrbnd = guard.lwrbnd
                    uprbnd = guard.uprbnd
                    var = guard.lhs
                    if guard.lwrbnd != -math.inf:
                        text += var + ' >= ' + str(lwrbnd) + ', ' # need to fix > vs >=
                    if guard.uprbnd != math.inf:
                        text += var + ' <= ' + str(uprbnd) + ' '

                text += '/'
                if len(inputs) > 0:
                    text += ' ?' + ', '.join(inputs)
                if len(outputs) > 0:
                    text += ' !' + ', '.join(outputs) 
                if len(internals) > 0:
                    text += ' #' + ', '.join(internals)
                
                automata.edge(q, qprime, label = text)

        return automata

def compose_components(component_1, component_2):
    new_component = ComponentAutomaton()
    newstart = (component_1.startState, component_2.startState)
    newtransitions = {}
    newstates = itertools.product(component_1.states, component_2.states)
    newends = itertools.product(component_1.endStates, component_2.endStates)
    newalphabet = component_1.alphabet.union(component_2.alphabet)
    dict1 = component_1.transitions_dict
    dict2 = component_2.transitions_dict
    for key1 in dict1:
        for key2 in dict2:
            for trans1 in dict1[key1]:
                for trans2 in dict2[key2]:
                    newtransitions[(key1, key2)] = compose_transitions(trans1, trans2)
    
    new_component.states = newstates
    new_component.endStates = newends
    new_component.startState = newstart

    new_component.transitions = newtransitions
    new_component.alphabet = newalphabet
    return new_component



class ContractAutomaton:
    def __init__(self):
        self.alphabet = {} # the alphabet for the contract
        self.transitions_dict = {} # dictionary of all transitions
        self.startState = None
        self.endStates = {}

    def add_state(self, name, transitions, end_state = 0):
        if end_state:
            self.endStates.append(name.upper())
        self.transitions_dict[name.upper()] = map(lambda x: x.upper(), transitions)

    def set_start_state(self, name):
        self.startState = name.upper()

    def simulate(self, is_printing = False):
        state = self.startState
        while state not in self.endStates:
            if is_printing:
                print(state + "  taking transitions, output, input")
            else:
                print("taking transitions, output, input")
            state, = sample(self.transitions_dict[state],1) # sample from set of all transistions uniformly at random
        print("simulation has terminated or deadlocked!")


# not 
# def merge_transition_dicts(td_1, td_2):
#     td_merged = td_1.copy()
#     for key in td_2:
#         if key in td_1:
#             td_merged[key] = td_1[key].intersection(td_2[key])
#         else:
#             td_merged[key] = td_2[key]

# def compose_contracts(contract_1, contract_2):
#     new_contract = ContractAutomaton()
#     if contract_1.startState == contract_2.startState:
#         new_contract.startState = contract_1.startState
#         new_contract.endStates = contract_2.endStates
#         new_contract.transitions_dict = merge_transition_dicts(contract_1.transitions_dict, contract_2.transitions_dict);
#     return new_contract

def refines_contracts(unrefined_c, refined_c):
    if unrefined_c.startState != refined_c.startState:
        return False
    elif unrefined_c.startState != refined_c.startState:
        return False
    else: 
        for key in unrefined_c.transitions_dict:
            if (key not in refined_c[key]):
                return False
            elif not refined_c[key].issubset(unrefined_c[key]):
                return False
            else:
                continue
        return True

contract_1 = ContractAutomaton()
