# Pedestrian Behaviors
# Tung M. Phan
# July 8th, 2018
# California Institute of Technology

from graphviz import Digraph


good_pedestrian = Digraph(filename='good_pedestrian.gv', comment='This is how a good pedestrian should behave')
good_pedestrian.node_attr.update(color='green', style='filled', fixedsize='true', width='1')

good_pedestrian.attr('node', shape = 'circle')
good_pedestrian.node ('1', 'crossed')
good_pedestrian.attr('node', shape = 'doublecircle')
good_pedestrian.node ('2', 'exit')
good_pedestrian.attr('node', shape = 'circle')
good_pedestrian.node ('0', 'start', color='yellow')

good_pedestrian.edge('0', '1', label='t_cross <= t_remaining / !cross')
good_pedestrian.edge('0', '0', label='⊤ / #wait')
good_pedestrian.edge('1', '1', label='⊤ / #wait')
good_pedestrian.edge('0', '2', label='⊤ / #walk')
good_pedestrian.edge('1', '2', label='⊤ / #walk')
good_pedestrian.view()

