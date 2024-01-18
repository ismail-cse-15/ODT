import simpy
from collections import deque
import pandas as pd
from config import GLOBAL




class StatCollector(object):
    def __init__(self, env: simpy.Environment, interval_minutes):
        self.env = env
        self.interval_minutes = interval_minutes
        self.individual_waiting_time_queue = deque()#[] #simpy.Store(env, capacity=int(interval_minutes))
        self.av_empty_time_queue = deque()#[] #simpy.Store(env, capacity=int(interval_minutes))
        self.av_idle_time_queue = deque()#[] #simpy.Store(env, capacity=int(interval_minutes))
        self.congestion_time_queue = deque()
        self.total_indiv_waiting = 0.0001
        self.total_av_empty = 0.0001
        self.total_av_idle = 0.0001
        self.total_congestion= 0.0001
        self.delta_indiv_waiting = 0
        self.delta_av_empty = 0
        self.delta_av_idle = 0
        self.delta_congestion = 0
        #self.dms_queue = simpy.Store(env, capacity=1)
        self.stat_df = pd.DataFrame(columns=['minutes', 'indiv-wait', 'av-empty', 'av-idle', 'congestion'], index=list(range(int(GLOBAL.RUN_UNTIL_HOUR*60/interval_minutes)+1)))
        self.stat_i = 0
        self.env.process(self.run())

    def set_data(self):
        pass

    # def collect_stat_old(self, dms: DynamicMobilityAssigner, now):
    #     indiv_wait = dms.get_total_indiv_waiting_time()
    #     av_empty = dms.get_total_av_empty_time()
    #     av_idle = dms.get_total_av_idle_time()
    #     row = [now, indiv_wait, av_empty, av_idle]
    #     self.stat_df.iloc[self.stat_i] = row
    #     self.stat_i += 1

    def process_queue(self, queue: deque, in_total: float):
        total = in_total
        delta = 0.0
        while len(queue) > 0:
            delta += queue.popleft()
        #total = 0.0001 if total < 0.0001 else total
        return total+delta, delta

    def collect_stat(self, now):
        self.total_indiv_waiting, self.delta_indiv_waiting = self.process_queue(self.individual_waiting_time_queue, self.total_indiv_waiting)
        self.total_av_empty, self.delta_av_empty = self.process_queue(self.av_empty_time_queue, self.total_av_empty)
        self.total_av_idle, self.delta_av_idle = self.process_queue(self.av_idle_time_queue, self.total_av_idle)
        self.total_congestion, self.delta_congestion = self.process_queue(self.congestion_time_queue, self.total_congestion)
        row = [now, self.total_indiv_waiting, self.total_av_empty, self.total_av_idle, self.total_congestion]
        self.stat_df.iloc[self.stat_i] = row
        self.stat_i += 1

    def run(self):
        #dms = yield self.dms_queue.get()
        #indiv_dic = indiv_av_list[0]
        #av_list = indiv_av_list[1]
        while True:
            yield self.env.timeout(self.interval_minutes)
            if self.env.now % 60 == 0:
                print("[Run-%d, AV-%d, MO-%s, GR-%d-%s, S-%d] StatCollector triggered at %f" % (GLOBAL.RUN_ID, GLOBAL.NUMBER_OF_AV,
                                                                                      GLOBAL.BALANCE_TYPE, GLOBAL.APPLY_GREEDY, GLOBAL.GREEDY_TYPE, GLOBAL.CAR_SEAT_CAPACITY, self.env.now))
            self.collect_stat(self.env.now)

            # if self.env.now > 24*60:
            #     break