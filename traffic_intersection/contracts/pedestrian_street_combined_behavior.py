# Pedestrian Behaviors
# Tung M. Phan
# July 8th, 2018
# California Institute of Technology

from graphviz import Digraph


combo = Digraph(filename='good_pedestrian_simple_lights.gv', comment='Pedestrian composed with traffic light')
combo.node_attr.update(color='green', style='filled', fixedsize='true', width='1.2')

combo.attr('node', shape = 'circle')
combo.node ('0', 'start, red', color='yellow')
combo.node ('1', 'start, green', color='yellow')
combo.node ('2', 'crossed, red')
combo.node ('3', 'crossed, green')
combo.attr('node', shape = 'doublecircle')
combo.node ('4', 'exit, red')
combo.node ('5', 'exit, green')

combo.edge('0', '1', label='⊤ / #wait, wait')
combo.edge('1', '1', label='⊤ / #wait, wait')
combo.edge('1', '0', label='⊤ / #wait, wait')
combo.edge('0', '0', label='⊤ / #wait, wait')
combo.edge('1', '2', label='t_walk == t_remaining / #cross')
combo.edge('1', '3', label='t_walk < t_remaining / #cross')
combo.edge('0', '4', label='⊤ / #walk, wait')
combo.edge('0', '5', label='⊤ / #walk, wait')
combo.edge('1', '4', label='⊤ / #walk, wait')
combo.edge('1', '5', label='⊤ / #walk, wait')
combo.edge('2', '3', label='⊤ / #wait, wait')
combo.edge('3', '2', label='⊤ / #wait, wait')
combo.edge('2', '4', label='⊤ / #wait, wait')
combo.edge('2', '5', label='⊤ / #wait, wait')
combo.edge('3', '4', label='⊤ / #wait, wait')
combo.edge('3', '5', label='⊤ / #wait, wait')
combo.view()

