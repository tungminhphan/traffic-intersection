# Simple Queue Class for Work Queue
# Tung M. Phan
# California Institute of Technology
# May 25th, 2018

class Queue():
    def __init__(self):
        self._queue = []

    def enqueue(self, element):
        self._queue.insert(0, element)

    def pop(self): 
        return self._queue.pop()

    def len(self):
        return len(self._queue)
