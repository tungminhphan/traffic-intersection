import automaton as a
import numpy as np


# would like to make it easier to test things lol
states1 = ['start', 'crossed', 'exit']
translist1 = {}
translist1[('start', 'crossed')] = ('3 ≤ x ≤ 5', {'yay'}, set(), set())
translist1[('start', 'start')] = ('y ≥ 14', set(), {'oh noes'}, set())
translist1[('crossed', 'exit')] = ('True', set(), set(), {'leave'})

c1 = a.construct_automaton(states1, translist1, 'start', {'exit'})

states2 = ['a', 'b']
translist2 = {}
translist2[('a', 'b')] = ('0 ≤ x ≤ 4', set(), {'yay'}, set())
translist2[('b', 'b')] = ('y ≤ 16', {'oh noes'}, set(), set())

c2 = a.construct_automaton(states2, translist2, 'a', {'b'})

# c1 = a.ComponentAutomaton()
# state1 = a.State('start')
# state2 = a.State('crossed')
# state3 = a.State('exit')

# c1.add_state(state1, 0, 1)
# c1.add_state(state2)
# c1.add_state(state3, 1)

# trans1 = a.guardTransition(state1, state2, a.iq.dictionarize(a.iq.Inequality('x', 3, 5)), {'yay'})
# trans2 = a.guardTransition(state1, state1, a.iq.dictionarize(a.iq.Inequality('y', 14)), set(), {'oh no'})
# trans3 = a.guardTransition(state2, state3, True, set(), set(), {'leave'})

# c1.add_transition(trans1)
# c1.add_transition(trans2)
# c1.add_transition(trans3)

# c1.convert_to_digraph().render('c1', view = True)

# c2 = a.ComponentAutomaton()
# state4 = a.State('a')
# state5 = a.State('b')

# c2.add_state(state4, 0, 1)
# c2.add_state(state5, 1)

# trans1 = a.guardTransition(state4, state5, a.iq.dictionarize(a.iq.Inequality('x', 0, 4)), set(), {'yay'})
# trans2 = a.guardTransition(state5, state5, a.iq.dictionarize(a.iq.Inequality('y', -np.inf, 16)), {'oh no'})

# c2.add_transition(trans1)
# c2.add_transition(trans2)

# c2.convert_to_digraph().render('c2', view = True)

c3 = a.compose_components(c1, c2)
c3.convert_to_digraph().render('c3', view = True)
