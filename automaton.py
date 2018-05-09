# Tung M. Phan
# California Institute of Technology
# May 8, 2018

class ContractAutomaton:

    def __init__(self):
        self.transitions = {} # implemented as a dict
        self.startState = None
        self.endStates = []

    def add_state(self, name, transition, end_state=0):
        if end_state:
            self.endStates.append(name)
        self.handlers[name] = handler

    def set_start_state(self, name):
        self.startState = name.upper()

    def simulate(self. name):
        state = self.startState
        while state not in self.endStates:
            continue


