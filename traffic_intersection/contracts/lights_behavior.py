# Traffic Light Behaviors
# Tung M. Phan
# July 8th, 2018
# California Institute of Technology

from graphviz import Digraph

simple_lights = Digraph(filename='simple_lights.gv', comment='This is a very simple traffic light model')
simple_lights.node_attr.update(color='green', style='filled', fixedsize='true', width='1')

simple_lights.attr('node', shape = 'circle')
simple_lights.node ('0', 'green', color='yellow')
simple_lights.node ('1', 'red', color='yellow')
simple_lights.attr('node', shape = 'circle')

simple_lights.edge('0', '1', label='t_walk == t_remaining / ?cross')
simple_lights.edge('0', '1', label='⊤ / #wait')
simple_lights.edge('0', '0', label='⊤ / #wait')
simple_lights.edge('1', '0', label='⊤ / #wait')
simple_lights.edge('1', '1', label='⊤ / #wait')
simple_lights.edge('0', '0', label='t_walk < t_remaining / ?cross')
simple_lights.view()

