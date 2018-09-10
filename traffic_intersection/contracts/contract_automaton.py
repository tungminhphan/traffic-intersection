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

