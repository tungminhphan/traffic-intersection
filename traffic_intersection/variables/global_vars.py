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
