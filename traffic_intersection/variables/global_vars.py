import sys
sys.path.append('..')
import prepare.queue as queue
current_time = 0
time_table = dict()
request_queue = queue.Queue()
waiting_dict = dict()
all_cars = dict()
effective_times = dict()
