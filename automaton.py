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
        self.endStates = []

    def add_state(self, name, transitions, end_state = 0):
        if end_state:
            self.endStates.append(name)
        self.transitions_dict[name] = transitions

    def set_start_state(self, name):
        self.startState = name.upper()

    def simulate(self):
        state = self.startState
        while state not in self.endStates:
            state, = random.sample(transistions[state],1) # sample from set of all transistions uniformly at random
            print("taking transitions, output, input")
        print("simulation has terminated or deadlocked!")



