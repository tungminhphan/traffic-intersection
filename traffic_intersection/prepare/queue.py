# Simple Queue Class for Work Queue
# Tung M. Phan
# California Institute of Technology
# May 25th, 2018

class Queue():
    def __init__(self):
        self._queue = list()

    def enqueue(self, element):
        '''
        insert new item at the back of the queue

        '''
        self._queue.insert(0, element)

    def insert_in_front(self, element):
        '''
        insert new item at the front of the queue

        '''
        self._queue.insert(-1, element)

    def pop(self): 
        '''
        remove and return the top element of the queue

        '''
        return self._queue.pop()

    def len(self):
        '''
        return the length of the queue

        '''
        return len(self._queue)

    def top(self):
        '''
        return the top element in the queue

        '''
        return self._queue[-1]

    def bottom(self):
        '''
        return the last element in the queue

        '''
        return self._queue[0]

    def replace_top(self, new_item):
        '''
        replace the top element in the queue by new_item

        '''
        self._queue[-1] = new_item

    def print(self):
        '''
        print the queue

        '''
        print(self._queue)

#prim_queue = Queue()
#prim_queue.enqueue((1, 0))
#prim_queue.enqueue((2, 0))
#print(prim_queue.top())
#prim_queue.print()
#print(prim_queue.bottom())
