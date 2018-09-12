from automaton import *

class ContractAutomaton(InterfaceAutomaton):
    def __init__(self, must = {}, may = {}):
        InterfaceAutomaton.__init__(self)
        self.must = must
        self.may = may
        self.transitions_dict = may
        # may and must are transition dictionaries

    def check_validity(self):
        # checks every must transition is a may transition
        for key in self.must:
            musttrans = self.must[key]
            maytrans = self.may[key]
            for must in musttrans:
                check = False
                for may in maytrans:
                    if check_simulation(must, may):
                        check = True

                if not check:
                    return False

        return True

    def add_transition(self, transition, must = 0):
        self.may[transition.startState].add(transition)
        if must:
            self.must[transition.startState].add(transition)

    def get_must_interface(self):
    	return InterfaceAutomaton(self.alphabet, self.must, self.startStates, self.endStates, self.failStates,
    		self.states, self.input_alphabet, self.output_alphabet, self.internal_alphabet)

    def get_may_interface(self):
		return InterfaceAutomaton(self.alphabet, self.may, self.startStates, self.endStates, self.failStates,
			self.states, self.input_alphabet, self.output_alphabet, self.internal_alphabet)

	def set_interface_automaton(self, interface):
		self.alphabet = interface.alphabet
		self.input_alphabet = interface.input_alphabet
		self.output_alphabet = interface.output_alphabet
		self.internal_alphabet = interface.internal_alphabet
		self.transitions_dict = interface.transitions_dict
		self.states = interface.states
		self.startStates = interface.startStates
		self.endStates = interface.endStates
		self.failStates = interface.failStates

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
            if state in self.startStates:
                automata.attr('node', color = 'yellow', fixedsize = 'true')
            newtext = ' ' * math.floor((maxlen - len(state.text))/2) + state.text + ' ' * math.ceil((maxlen - len(state.text))/2)
            automata.node(newtext, newtext)

        # adds transitions
        for state in self.states:
            maytransit = self.may[state] # this is a set of may transitions from the state
            musttransit = self.must[state]
            for trans in maytransit:
                if trans is not False:
                    state2 = trans.endState
                    transtext = trans.show()
                    automata.edge(state.text, state2.text, label = transtext, style = 'dotted')

            for trans in musttransit:
                if trans is not False:
                    state2 = trans.endState
                    transtext = trans.show()
                    automata.edge(state.text, state2.text, label = transtext)

        return automata

    def prune_illegal_state(self):
    	# remove any states such that must does not imply may
        finished = False
        while not finished:
            finished = True
            for key in self.must:
                musttrans = must[key]
                maytrans = may[key]
                for must in musttrans:
                    check = False
                    for may in maytrans:
                        if check_simulation(must, may):
                            check = True

                    if not check:
                        self.remove_state(key)
                        finished = False


    def weakAlphabetProjection(self, contract):
    	# adds may self-loops
        alphabetDifference = contract.alphabet - self.alphabet
        for state in self.states:
            for letter in alphabetDifference:
                selfloop = guardTransition(state, state, 'True', letter, '#')
                self.add_transition(selfloop, 0)

    def strongAlphabetProjection(self, contract):
    	# adds must self-loops
        alphabetDifference = contract.alphabet - self.alphabet
        for state in self.states:
            for letter in alphabetDifference:
                selfloop = guardTransition(state, state, 'True', letter, '#')
                self.add_transition(selfloop, 1)

def compose_contract(cr_1, cr_2):
	cr_1 = cr_1.strongAlphabetProjection(cr_2)
	cr_2 = cr_2.strongAlphabetProjection(cr_1)

	must1 = cr_1.get_must_interface()
	must2 = cr_2.get_must_interface()
	may1 = cr_1.get_may_interface()
	may2 = cr_2.get_may_interface()

	mustAuto = compose_interface(must1, must2)
	mayAuto = compose_interface(may1, may2)

	# TODO: Add in pruning for composition

	contract = ContractAutomaton(must = mustAuto.transitions_dict, may = mayAuto.transitions_dict)
	contract.set_interface_automaton(mayAuto)
	return contract

def check_simulation(trans1, trans2):
    # checks if trans1 <= trans2, ie they have the same action, action type, and g_1 => g_2
    # TODO


# assumes weight on a graph is a string of the form "guard / ?input, not output, #internal separated by , "
def convert_graph_to_automaton(digraph):
    new_interface = InterfaceAutomaton()
    nodes = digraph._nodes
    edges = digraph._edges
    stringstatedict = {}
    inp = set()
    out = set()
    inter = set()

    for node in nodes:
        newstate = State(node)
        new_interface.add_state(node)
        stringstatedict[node] = newstate

    for source in sources:
        new_interface.set_start_state(stringstatedict[source])

    for sink in sinks:
        new_interface.endStates.add(stringstatedict[sink])


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

        # Each action in form ?(input), not (output), or #(internal)
        for action in actions:
            if action[0] == '?':
                inp.add(action[1:])
            elif action[0] == 'not ':
                out.add(action[1:])
            elif action[0] == '#':
                inter.add(action[1:])
            else:
                raise SyntaxError('Input, output or internals in wrong format.')

        new_interface.add_transition(guardTransition(state1, state2, guard, inp, out, inter))

    return new_interface

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

