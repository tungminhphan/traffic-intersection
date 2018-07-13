# Alternative Contract Automaton Class
# Tung M. Phan
# California Institute of Technology
# July 12, 2018

from random import sample
import math
import itertools
import numpy as np

# Inequality for guard. Left-hand side is variable name, and is bounded on both sides (if one-sided inequality, has +/- infty on other side.)
class Inequality:
    def __init__(self, var, lower = -math.inf, upper = math.inf):
        self.var = var # variable name
        self.lwrbnd = lower # the lower bound
        self.uprbnd = upper # the upper bound

    def show(self):
        if self.lwrbnd > self.uprbnd:
            txt = False
        elif self.lwrbnd == -math.inf and self.uprbnd == math.inf:
            txt = True
        elif self.lwrbnd == self.uprbnd:
            txt = self.var + " = " + str(self.lwrbnd)
        elif self.lwrbnd == -math.inf and self.uprbnd != math.inf:
            txt = self.var + ' ≤ ' + str(self.uprbnd)
        elif self.lwrbnd != -math.inf and self.uprbnd == math.inf:
            txt = str(self.lwrbnd) + ' ≤ ' + self.var
        elif self.lwrbnd != -math.inf and self.uprbnd == math.inf:
            txt = str(self.lwrbnd) + ' ≤ ' + self.var
        else:
            txt = str(self.lwrbnd) + ' ≤ ' + self.var + ' ≤ ' + str(self.uprbnd)
        return txt

# returns a set of inequalities whose conjunction has the same truth value as the conjunction of two sets of inequalities
def conjunct(ineq_dict1, ineq_dict2):
    keys1 = set(ineq_dict1.keys())
    keys2 = set(ineq_dict2.keys())

    shared_keys = keys1 & keys2
    different_keys = keys1 | keys2 - keys1 & keys2
    new_dict = dict()

    for key in shared_keys:
        ineq1 = ineq_dict1[key]
        ineq2 = ineq_dict2[key]
        new_ineq = Inequality(ineq1.var, min(ineq1.lwrbnd, ineq2.lwrbnd), max(ineq1.uprbnd,ineq2.uprbnd)) # take the conjunction of the two inqualities
        if new_ineq.show() == False:
            return False
        elif new_ineq.show() != True:
            new_dict[key] = new_ineq

    for key in different_keys:
        if key in ineq_dict1.keys():
            new_dict[key] = ineq_dict1[key]
        else:
            new_dict[key] = ineq_dict2[key]
    return new_dict

def dictionarize(ineq):
    ineq_dict = dict()
    ineq_dict[ineq.var] = ineq
    return ineq_dict

def pretty_print(ineq_dict): # print contents of a dictionary of inequalities
    keys = sorted(ineq_dict.keys())
    for key in keys:
        print(ineq_dict[key].show())

# TODO: fix comments returns a set of inequalities that has the same truth value as the disjunction of two sets of inequalities# returns an inequality that has the same truth value as disjunction of two inequalities
def disjunct(ineq1, ineq2):
    if ineq1.var == ineq2.var:
        return Inequality(guard1.var, min(ineq1.lwrbnd, ineq2.lwrbnd), max(ineq1.uprbnd, ineq2.uprbnd))
    else:
        return False # ? don't need to ever use this in this case

# returns an inequality that has the same truth value as the conjunction of two inequalities (or if
# the two inequalites correspond to different variables, a dictionary

eq1 = dictionarize(Inequality(lower = 3, var = 'x', upper = 3))
eq2 = dictionarize(Inequality(lower = -9, var = 'z', upper = 5))
eq3 = dictionarize(Inequality(lower = 2, var = 'x', upper = 7))
eq4 = dictionarize(Inequality(lower = -3, var = 'z', upper = 10))
eq =  conjunct(conjunct(eq1,eq2), conjunct(eq3,eq4))
pretty_print(eq)

# Transition class. Has guard and optional input/output/internal action
class Transition:
    def __init__(self, ineq = True, inp = None, out = None, inter = None, s1 = None, s2 = None):
        self.guard = ineq # guard is a set of inequalities joined by conjunction (?)
        self.input = inp # set of inputs
        self.output = out
        self.internal = inter # internal action?
        self.state1 = s1 # set
        self.state2 = s2 # transition from s1 to s2

def compose_transitions(tr1, tr2):
    guard = conjunct(tr1, tr2)
    inp = tr1.input.union(tr2.input)
    inter = (tr1.input.intersection(tr2.output)).union(tr1.output.intersection(tr1.input))
    out = tr1.output.intersection(tr2.output)
    return Transition(guard, inp, out, inter, (tr1.state1, tr2.state1), (tr1.state2, tr2.state2))


class ComponentAutomaton:
    def __init__(self):
        self.alphabet = {} # the alphabet for the component
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

    def compose_components(component_1, component_2):
        new_component = ComponentAutomaton()
        newstart = (component_1.startState, component_2.startState)
        newtransitions = []
        newends = itertools.product(component_1.endStates, component_2.endStates)
        dict1 = component_1.transitions_dict
        dict2 = component_2.transitions_dict
        for key1 in dict1:
            for key2 in dict2:
                newtransitions.append(compose_transitions(dict1[key1], dict2[key2]))
        return newtransitions

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
