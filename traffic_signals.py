# Traffic Light Class
# Tung M. Phan
# May 31st, 2018
# California Institute of Technology

yellow_duration = 10
import random

class TrafficLight():
    def __init__(self, yellow_max = 5, green_max = 25, random_start = True):
        self._yellow_max = yellow_max
        self._green_max = green_max
        self._red_max = self._yellow_max + self._green_max
        if random_start:
            colors = ['red', 'yellow', 'green']
            self._horizontal = [random.choice(colors), 0]
            self._vertical = [random.choice(colors), 0]
        else:
            self._horizontal = ['red', 0]
            self._vertical = ['green', 0]




traffic_light = TrafficLight(yellow_max = 3, green_max = 30)


print(traffic_light._red_max)
print(traffic_light._horizontal)


