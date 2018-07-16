from automaton import ComponentAutomaton, Inequality, Transition, compose_components
import math

simple = ComponentAutomaton()
simple.set_start_state('start')
s_transitions = [Transition('start', 'crossed', Inequality('x', 3, 5), ['yay']), 
Transition('start', 'start', Inequality('y', 14), [], ['oh no'])]
simple.add_state('start', s_transitions)
e_transitions = [Transition('crossed', 'exit', True, [], [], ['leave'])]
simple.add_state('crossed', e_transitions)
simple.add_state('exit', [], 1)
simple.convert_component().render('c1', view = True)

simple2 = ComponentAutomaton()
simple2.set_start_state('a')
s_transitions = [Transition('a', 'b', Inequality('x', 0, 4), [], ['yay']), 
Transition('a', 'a', Inequality('y', 5, math.inf), ['oh no'], [])]
simple2.add_state('a', s_transitions)
e_transitions = [Transition('b', 'b', True, [], [], ['yo'])]
simple2.add_state('b', e_transitions, 1)
simple2.convert_component().render('c2', view = True)

composition = compose_components(simple, simple2)
composition.convert_component().render('c3', view = True)
