# Contract Automaton Class
# Tung M. Phan
# California Institute of Technology
# May 8, 2018



from random import sample

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
