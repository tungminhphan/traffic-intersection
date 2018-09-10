import automaton

class ContractAutomaton(Automaton):
    def __init__(self, must = {}, may = {}):
        Automaton.__init__(self)
        self.must = must
        self.may = may
        # may and must are transition dictionaries

    def check_validity(self):
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
                if state in self.startStates:
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
                , key1 in c_1.startStates and key2 in c_2.startStates)

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

# assumes weight on a graph is a string of the form "guard / ?input, !output, #internal separated by , "
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

        new_interface.add_transition(guardTransition(state1, state2, guard, inp, out, inter))

    return new_interface

