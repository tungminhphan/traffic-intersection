# Simple Directed Graph Class for Waypoint Graph
# Tung M. Phan
# California Institute of Technology
# May 9th, 2018

class DirectedGraph():
    def __init__(self):
        self._nodes = set() # set of nodes
        self._edges = {} # set of edges is a dictionary of sets (of nodes)

    def add_node(self, node): # add a node
            self._nodes.add(node)

    def add_edges(self, edge_set): # add edges
        for edge in edge_set:
            for node in edge:
                if node not in self._nodes:
                    self.add_node(node)
            try: self._edges[edge[0]].add(edge[1])
            except KeyError:
                self._edges[edge[0]] = {edge[1]}

    def print_graph(self):
        print('The directed graph has ' + str(len(self._nodes)) + ' nodes: ')
        print(str(list(self._nodes)).strip('[]'))
        print('and ' + str(sum([len(self._edges[key]) for key in self._edges])) + ' edges: ')
        for start_node in self._edges:
            print(str(start_node) + ' -> ' +  str(list(self._edges[start_node])).strip('[]'))

