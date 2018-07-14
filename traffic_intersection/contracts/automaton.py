# Contract Automaton Class
# Tung M. Phan
# California Institute of Technology
# May 8, 2018



from random import sample
import math
import itertools


# Inequality for guard. Left-hand side is variable name, and is bounded on both sides (if one-sided inequality, has +/- infty on other side.)
class Inequality:
    def __init__(self, left = '', lower = -math.inf, upper = math.inf):
        self.lhs = left # a variable
        self.lwrbnd = lower # a number
        self.uprbnd = upper 

# returns an inequality that has the same truth value as the conjunction of two inequalities
def conjunct(ineq1, ineq2):
    if ineq1.lhs == ineq2.lhs:
        return Inequality(guard1.lhs, max(ineq1.lwrbnd, ineq2.lwrbnd), min(ineq1.uprbnd, ineq2.uprbnd))
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
            if ineq1.lhs == ineq2.rhs:
                set3.append(conjunct(ineq1, ineq2))
    return set3

# returns a set of inequalities that has the same truth value as the disjunction of two sets of inequalities
def disjunctset(set1, set2):
    return True

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

## TESTING
contract_1 = ContractAutomaton()
ineq = Inequality(left = 'x')
