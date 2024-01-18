import simpy
import numpy as np
from scipy.spatial.distance import cdist, cosine
import networkx as nx
from config import GLOBAL, get_link_length, nx_shortest_path, calculate_link_capacity, extend_by_shortest_path, get_shortest_travel_time
from mop import AssessAVsToServeTrip
from agents import TripRequest, AutonomousVehicle, Individual, TripUpdate
import pandas as pd
import matplotlib.pyplot as plt
from statistics import StatCollector
import logging
from datetime import datetime
from typing import List
from collections import deque


class SimulatedWorld(object):
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.indiv_dic = {}
        self.av_list = []
        self.stat_collector = StatCollector(env, 60)

        df_survey = GLOBAL.df_survey.head(GLOBAL.NUMBER_OF_TRIPS)
        self.generate_link_capacity()
        for i in range(GLOBAL.NUMBER_OF_AV):
            av = AutonomousVehicle(i, env, len(df_survey), GLOBAL.NODES_LIST[np.random.randint(len(GLOBAL.NODES_LIST))], self.stat_collector, GLOBAL.CAR_SEAT_CAPACITY)
            self.av_list.append(av)
        dms = DynamicMobilityAssigner(self.env, len(df_survey), self.av_list, self.stat_collector)
        person_id = set(df_survey['indiv_id'].to_list())
        for indiv in person_id:
            activity_df = df_survey[df_survey['indiv_id'] == indiv]
            activity_df = activity_df.sort_values(by='origin_departure_time', ignore_index=True)
            indiv_obj = Individual(int(indiv), self.env, dms, activity_df, self.stat_collector)
            self.indiv_dic[int(indiv)] = indiv_obj
        dms.set_indiv(self.indiv_dic)


    def generate_link_capacity(self):
        for (i, j) in GLOBAL.G.edges:
            if i < j:
                link_length = get_link_length(i, j)
                cap = calculate_link_capacity(link_length)
                GLOBAL.LINK_CAPACITY[(i, j)] = simpy.Resource(self.env, cap)
                GLOBAL.LINK_CAPACITY[(j, i)] = simpy.Resource(self.env, cap)


class DynamicMobilityAssigner(object):
    def __init__(self, env: simpy.Environment, max_request: int, av_list: List[AutonomousVehicle], stat_collector: StatCollector):
        self.env = env
        self.max_request = max_request
        self.dms_request_queue = simpy.Store(self.env, capacity=max_request)
        self.av_list = av_list#[]
        self.last_selected_av = 0
        self.indiv_dic = None
        self.mo_assess_av = AssessAVsToServeTrip(len(av_list), GLOBAL.NUMBER_OF_OBJECTIVES_ASSIGN_TRIP, GLOBAL.NUMBER_OF_THREADS)
        self.last_total_indiv_waiting_time_growth = 0.5
        self.last_total_av_empty_time_growth = 0.5
        self.stat_collector = stat_collector
        self.env.process(self.run())

    def run(self):
        while True:
            trip_req = yield self.dms_request_queue.get()
            av_i, path = self.assess_avs_for_current_trip(trip_req)
            if av_i == -1:
                yield trip_req.indiv_travel_queue.put(TripUpdate(-1, None, None, 'fail', -1))
                continue

            route = extend_by_shortest_path(path)
            self.av_list[av_i].assign_trip(trip_req, route)

            if self.av_list[av_i].is_idle:
                self.av_list[av_i].agent_process.interrupt()


    def set_indiv(self, indiv_dic):
        self.indiv_dic = indiv_dic

    def get_total_indiv_waiting_time(self):
        return (self.stat_collector.delta_indiv_waiting +
                self.last_total_indiv_waiting_time_growth)/self.stat_collector.total_indiv_waiting


    def get_total_av_empty_time(self):
        return (self.stat_collector.delta_av_empty +
                self.last_total_av_empty_time_growth) /self.stat_collector.total_av_empty
        #return self.stat_collector.total_av_empty

    def get_total_av_idle_time(self):
        return (self.stat_collector.delta_av_idle)/self.stat_collector.total_av_idle
        #return self.stat_collector.total_av_idle

    def get_total_av_cost(self):
        return (self.stat_collector.delta_av_empty + self.stat_collector.delta_av_idle + self.last_total_av_empty_time_growth)\
        /(self.stat_collector.total_av_empty + self.stat_collector.total_av_idle)

    def select_decision_vector(self):
        vec_indiv_av = [0.5, 0.5]
        if GLOBAL.BALANCE_TYPE == 'INDIV':
            vec_indiv_av = [0.99, 0.01]
        elif GLOBAL.BALANCE_TYPE == 'AV':
            vec_indiv_av = [0.01, 0.99]
        elif GLOBAL.BALANCE_TYPE == 'RND':
            r = np.random.rand()
            vec_indiv_av[0] = r
            vec_indiv_av[1] = 1 - r
        elif GLOBAL.BALANCE_TYPE == 'ADAPT_OLD':
            current_state = np.array([self.get_total_indiv_waiting_time(), self.get_total_av_cost()])
            decision_vector = current_state / (np.sum(current_state) + 0.0001)
            vec_indiv_av = [decision_vector[0] + 0.0001, decision_vector[1]+ 0.0001]
        elif GLOBAL.BALANCE_TYPE == 'ADAPT_OLD_MIRROR':
            current_state = np.array([self.get_total_indiv_waiting_time(), self.get_total_av_cost()])
            decision_vector = current_state / (np.sum(current_state) + 0.0001)
            vec_indiv_av = [1-decision_vector[0], 1-decision_vector[1]]
        return np.array(vec_indiv_av)
        

    def adjust_indiv_av_state(self, x, d, t):
        if GLOBAL.BALANCE_TYPE == 'ADAPT':
            x = x + d
            x = x / t
        return x

    def assess_by_cosine_distance(self, indiv_av_state):
        if len(indiv_av_state) == 1:
            return 0

        decision_vector = self.select_decision_vector() #1.0 - decision_vector
        cos_score_to_be_min = np.zeros(len(indiv_av_state))
        len_to_be_min = np.zeros(len(indiv_av_state))
        d = np.array([self.stat_collector.delta_indiv_waiting, self.stat_collector.delta_av_empty + self.stat_collector.delta_av_idle])
        t = np.array([self.stat_collector.total_indiv_waiting, self.stat_collector.total_av_empty + self.stat_collector.total_av_idle])
        for i in range(len(indiv_av_state)): #todo: 2D pareto sort can be used here also
            x = np.array(indiv_av_state[i])
            x = x + 0.0001 # to avoid dealing zero vector
            x = self.adjust_indiv_av_state(x, d, t)
            cos_distance = cosine(decision_vector, x)
            if cos_distance == 0.0:
                cos_distance = 0.0001
            cos_score_to_be_min[i] = cos_distance #* np.sqrt(x.dot(x)) #adjusted by norm to evade worst case
            len_to_be_min[i] = 1/np.sqrt(x.dot(x))
        cos_score_to_be_min = cos_score_to_be_min/cos_score_to_be_min.sum()
        len_to_be_min = len_to_be_min/len_to_be_min.sum()
        cos_score_to_be_min = cos_score_to_be_min * len_to_be_min
        cos_score_to_be_min = cos_score_to_be_min/cos_score_to_be_min.sum()
        if GLOBAL.BALANCE_TYPE == 'ADAPT_OLD':
            min_index = np.random.choice(len(indiv_av_state), 1, p=cos_score_to_be_min)
            return min_index[0]
        else:
            min_index = np.where(cos_score_to_be_min == cos_score_to_be_min.max())[0]
            return min_index[np.random.randint(0, len(min_index))]
        # min_index = np.where(np.abs(cos_score_to_be_min - cos_score_to_be_min.max()) < 0.01)[0]
        # min_len = np.min(len_to_be_min[min_index])
        # min_len_index = np.where(np.abs(len_to_be_min - min_len) < 0.01)[0]
        # if GLOBAL.VERBOSE:
        #     if len(min_index) > 1:
        #         print("No of equivalent AVs: " + str(len(min_index)))

        # return min_len_index[np.random.randint(0, len(min_len_index))]

    def assess_nd_avs_for_current_trip(self, nd_avs: list, trip: TripRequest): #L2 filter: considers global aspect
        #av_lag_demand_supply = []
        indiv_av_state = []
        fetch_route = {}
        # if GLOBAL.VERBOSE:
        #     print("No of nd_avs: " + str(len(nd_avs)))
        #for each nd-av
        for av_id in nd_avs:
            av_start_node = self.av_list[av_id].current_node if self.av_list[av_id].current_destination is None else self.av_list[av_id].current_destination
            if len(self.av_list[av_id].trip_queue) > 0:
                av_start_node = self.av_list[av_id].trip_queue[-1].destination
            #calculate best routes to fetch indiv
            route = nx_shortest_path(source=av_start_node, target=trip.origin, weight='length')
            fetch_route[av_id] = route
            # #calculate actual indiv waiting time and empty trip duration
            # empty_time = get_shortest_travel_time_route(route)#get_travel_time_minutes(route)
            # indiv_waiting_time = empty_time
            # if self.av_list[av_id].current_destination is not None:
            #     time_to_complete_prev_trip = get_shortest_travel_time_route(self.av_list[av_id].current_route, start_node=self.av_list[av_id].current_node)
            #     #get_travel_time_minutes(self.av_list[av_id].current_route, self.av_list[av_id].current_route.index(self.av_list[av_id].current_node))
            #     for assigned_trip in self.av_list[av_id].trip_queue.items:
            #         time_to_complete_prev_trip += get_shortest_travel_time_route(assigned_trip.route)#get_travel_time_minutes(assigned_trip.route)
            #     indiv_waiting_time += time_to_complete_prev_trip
            # idle_time = self.av_list[av_id].compute_current_idle_time(trip.time)
            indiv_waiting_time = self.av_list[av_id].obj_scores['indiv_wait']
            empty_time = self.av_list[av_id].obj_scores['empty']
            idle_time = self.av_list[av_id].obj_scores['idle']
            indiv_av_state.append([indiv_waiting_time, empty_time + idle_time]) #todo: should we normalize here also?
        return indiv_av_state, fetch_route

    def update_stats_based_on_selected_av(self, trip: TripRequest, fetch_route, indiv_av_state):
        if GLOBAL.VERBOSE:
            if indiv_av_state[0] > 30:
                print('Trip assignment cause indiv waiting: ' + str(indiv_av_state[0]))
        trip.set_route(fetch_route)
        # total_indiv_waiting_time = 1#self.get_total_indiv_waiting_time()
        # total_av_empty_time = 1#self.get_total_av_empty_time()
        #total_av_idle_time = self.get_total_av_idle_time()
        self.last_total_indiv_waiting_time_growth = indiv_av_state[0]#(indiv_av_state[0] + total_indiv_waiting_time) / total_indiv_waiting_time
        self.last_total_av_empty_time_growth = indiv_av_state[1]#(indiv_av_state[1] + total_av_empty_time )/ total_av_empty_time


    def select_from_nd_avs(self, nd_avs: list, trip: TripRequest):
        indiv_av_state, fetch_route = self.assess_nd_avs_for_current_trip(nd_avs, trip)
        selected_i = self.assess_by_cosine_distance(indiv_av_state)
        selected_av = nd_avs[selected_i]
        self.update_stats_based_on_selected_av(trip, fetch_route[selected_av], indiv_av_state[selected_i])
        # trip.set_fetch_routes(fetch_route[selected_av])
        # total_indiv_waiting_time = self.get_total_indiv_waiting_time()
        # total_av_empty_time = self.get_total_av_empty_time()
        # self.last_total_indiv_waiting_time_growth = (indiv_av_state[selected_i][0] + total_indiv_waiting_time) / total_indiv_waiting_time
        # self.last_total_av_empty_time_growth = (indiv_av_state[selected_i][1] + total_av_empty_time )/total_av_empty_time
        return selected_av

    def assess_avs_for_current_trip(self, trip: TripRequest): #L1 filter: consider local aspects
        potential_av_ids = self.filter_avs(trip)
        self.mo_assess_av.set_data(trip, self.av_list, self.indiv_dic[trip.individual])
        return self.mo_assess_av.get_nd_indices_greedy(potential_av_ids)


    def select_av(self):
        self.last_selected_av = (self.last_selected_av + 1) % len(self.av_list)
        return self.av_list[self.last_selected_av]


    # def serve_individual(self, av_travel_queue, origin: int, destination: int, indiv: int):
    #     trip_req = TripRequest(origin, destination, av_travel_queue, indiv)
    #     #### experimental #######
    #     #av = self.launch_new_av(origin)
    #     av_i = self.assess_avs_for_current_trip(trip_req)
    #     #av = self.select_av()
    #     self.av_list[av_i].assign_trip(trip_req)

    def launch_new_av(self, origin: int):
        av = AutonomousVehicle(DynamicMobilityAssigner.av_count, self.env, self.max_request, origin)
        self.av_list.append(av)
        DynamicMobilityAssigner.av_count += 1
        return av

    def filter_avs(self, trip: TripRequest):
        potential_av_ids = []

        for av in self.av_list:
            if av.is_idle:
                if self.env.now > av.idle_start:
                    av.track_idle()
            if get_shortest_travel_time(av.current_node, trip.origin) > GLOBAL.INDIV_MAX_WAIT_MIN:
                if not av.is_idle:
                    continue
            if len(av.id_to_trip) == GLOBAL.CAR_SEAT_CAPACITY:
                continue
            potential_av_ids.append(av.id_num)
        return potential_av_ids



if __name__ == '__main__':
    seeds = [24204460, 255701664, 255452145, 243420000, 233962792, 70460023, 164387811, 95284976, 248008688, 236908554,
             250416188, 60094985, 28918358, 52010314, 261135298, 14898472, 86355748, 266651191, 90120307, 166684931]
    np.random.seed(seeds[0])
    save_dir="../figure/"
    msg = GLOBAL.BALANCE_TYPE + '-av-' + str(GLOBAL.NUMBER_OF_AV)

    today = datetime.now()
    timestamp = today.strftime("%d-%m-%y %H:%M")
    logging.basicConfig(filename='../log/' + timestamp + msg + '.log', level=logging.INFO, filemode='w', format='%(levelname)s;%(name)s;%(message)s')
    logging.info('TIME;MSG;INDIV;AV;TRIP;NODE;OTHER;NOW;NUM_PEOPLE')
    env = simpy.Environment(initial_time=GLOBAL.RUN_START_HOUR*60) #initial_time=5*60
    world = SimulatedWorld(env)
    env.run(until=(GLOBAL.RUN_UNTIL_HOUR+0.001)*60)
    stat = world.stat_collector.stat_df
    #plt.plot(stat['minutes'], stat['indiv-wait'])
    #plt.plot(stat['minutes'], stat['indiv-wait'])
    stat.plot.line('minutes', [ 'indiv-wait', 'av-empty', 'av-idle']) #, 'av-idle'
    plt.yscale('log')
    plt.savefig(save_dir+"time_indiv-wait_av-empty_av-idle.pdf")
    plt.show()
