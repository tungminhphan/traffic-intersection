import sys
sys.path.append('..')
import prepare.queue as queue
current_time = 0
all_cars = dict()
cars_to_remove = set()

curr_tubes = []
prim_ids_to_show = []
walls = []
ids = []
boxes = []
honk_waves = []
all_wavefronts = set()
pedestrians_waiting = set()
pedestrians_to_keep= set()
cars_to_show = []
pedestrians_to_show = []
crossing_highlights = []
show_traffic_lights = []
