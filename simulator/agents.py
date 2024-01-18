import simpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from pyproj import Proj#/Volumes/Nur/Nur/Phd@CSE,BUET/DAL/data/
from scipy.spatial.distance import cdist
import pickle
import networkx as nx
from config import GLOBAL, get_link_length, get_x_y, log_event, nx_shortest_path
import logging
import datetime
from statistics import StatCollector
from collections import deque
from typing import List, Dict



completed_trip=0
class TripRequest(object):
    def __init__(self, origin: int, dest: int, indiv_travel_queue: simpy.Store, indiv: int, trip_id: int, time=None, arrival_time=None, num_people=1): #, fetch_route=None):
        self.origin = origin
        self.destination = dest
        self.indiv_travel_queue = indiv_travel_queue
        self.individual = indiv
        self.trip_id = trip_id
        self.time = time
        self.exp_arrival_time = arrival_time
        self.num_people = num_people
        self.pickup_time = None
        #self.route = route
        #self.fetch_route = fetch_route

    def set_route(self, route):
        self.route = route


class TripUpdate(object):
    def __init__(self, node, x, y, state: str, av_id):
        self.current_node = node
        self.current_x = x
        self.current_y = y
        self.state = state #fetch, start, progress, end, fail
        self.av_id = av_id


class Individual(object):
    def __init__(self, id_num: int, env: simpy.Environment, dms, activity_df: pd.DataFrame, stat_collector: StatCollector):
        self.id_num = id_num
        self.env = env
        self.activity_queue = simpy.Store(self.env, capacity=len(activity_df))
        for index, row in activity_df.iterrows(): #for index, row in df.iterrows():
            self.activity_queue.put(row.to_dict()) #todo: do we need yield
        self.av_travel_queue = simpy.Store(self.env, capacity=GLOBAL.G.size())
        self.dms = dms
        self.current_node = None
        self.current_x = None
        self.current_y = None
        self.wait_start_time = None
        self.last_wait_time = None
        self.total_wait_time = 0.0001
        self.stat_collector = stat_collector
        self.env.process(self.run())

    def run(self):
        while len(self.activity_queue.items) > 0:
            activity = yield self.activity_queue.get()
            trip_id = activity['trip_id']
            self.update_location(activity['origin_node'], None, None, trip_id)
            next_trip_time = activity['origin_departure_time']
            next_trip_time = 60 * int(next_trip_time/100) + next_trip_time % 100
            next_trip_arrival_time = activity['destination_arrival_time']
            next_trip_arrival_time = 60 * int(next_trip_arrival_time/100) + next_trip_arrival_time % 100 + GLOBAL.ARRIVAL_TIME_SLACK_MINUTES
            time_left = next_trip_time - self.env.now
            if time_left < 0:
                time_left = 0
            yield self.env.timeout(time_left)
            #logging.info('Inividual-%d requested trip-%d at %s to travel from node-%d to node-%d' % (self.id_num, trip_id,
            #                                                                          str(datetime.timedelta(minutes=self.env.now)),
            #                                                                          activity['origin_node'], activity['destination_node']))
            log_event(self.env.now, 'TRIP_REQUEST', indiv_i=self.id_num, trip_i=trip_id, node=activity['origin_node'],
                      period=activity['destination_node'])
            self.wait_start_time = self.env.now
            trip = TripRequest(activity['origin_node'], activity['destination_node'], self.av_travel_queue, self.id_num, trip_id,
                               time=self.wait_start_time, arrival_time=next_trip_arrival_time)
            #self.dms.serve_individual(self.av_travel_queue, activity['origin_node'], activity['destination_node'], self.id_num)
            yield self.dms.dms_request_queue.put(trip)

            while True:
                trip_update = yield self.av_travel_queue.get() #get from self.av_travel_queue
                if trip_update.state == "start":  #interpret state
                    self.last_wait_time = self.env.now - self.wait_start_time #if start travel calculate waiting time
                    self.total_wait_time += self.last_wait_time
                    self.stat_collector.individual_waiting_time_queue.append(self.last_wait_time)
                    #logging.info('Inividual-%d waited for %f min for trip-%d at node-%d' % (self.id_num, self.last_wait_time,
                    #                                                                        trip_id, self.current_node))
                    log_event(self.env.now, 'INDIV_WAIT', indiv_i=self.id_num, av_i=trip_update.av_id, trip_i=trip_id, node=self.current_node, period=self.last_wait_time)
                # elif trip_update.state == "progress": #else if edge passed update location
                #     self.update_location(trip_update.current_node, trip_update.current_x, trip_update.current_y, trip_id)
                elif trip_update.state == "end": #else if end travel break
                    self.update_location(trip_update.current_node, trip_update.current_x, trip_update.current_y, trip_id )
                    log_event(self.env.now, 'INDIV_ARRIVAL_DELAY', indiv_i=self.id_num, av_i=trip_update.av_id, trip_i=trip_id, node=self.current_node, period=self.env.now-trip.exp_arrival_time)
                    #todo: stat_collector
                    break
                elif trip_update.state == "fail":
                    log_event(self.env.now, 'REQ_FAILED', indiv_i=self.id_num, trip_i=trip_id, node=self.current_node)
                    break


    def assign_trip(self, trip: TripRequest):
        self.activity_queue.put(trip)

    def update_location(self, node, x, y, trip_id):
        self.current_node = node
        self.current_x = x
        self.current_y = y
        #logging.debug('Inividual-%d trip-%d now on node-%d (%.4f, %.4f) at %s' % (self.id_num, trip_id, self.current_node,
        #                                                          self.current_x, self.current_y, str(datetime.timedelta(minutes=self.env.now)) ))
        # print('Inividual-%d now on node-%d (%.4f, %.4f) at %d' % (self.id_num, self.current_node, self.current_x,
        # self.current_y, self.env.now))


class AutonomousVehicle(object):
    def __init__(self, id_num: int, env: simpy.Environment, max_request: int, starting_node: int, stat_collector: StatCollector, seat_capacity):
        self.id_num = id_num
        self.env = env
        self.trip_queue = deque()#simpy.Store(self.env, capacity=max_request)
        self.current_node = starting_node
        self.current_destination = None
        self.last_trip_start_time = None
        self.last_trip_end_time = None
        self.obj_scores = {'empty': 0, 'idle': 0, 'indiv_wait': 0}
        self.trip_started = False
        self.trip_assigned = False
        self.idle_start = self.env.now
        #self.idle_end = None
        self.total_idle = 0.0001
        self.empty_trip_start = None
        self.empty_trip_end = None
        self.total_empty_trip = 0.0001
        self.total_effective_distance = 0.0001
        self.current_route = None
        self.stat_collector = stat_collector
        self.is_idle = True
        self.traverse_process = None
        self.seat_capacity = seat_capacity
        self.next_id_to_fill = 0 #GLOBAL.CAR_SEAT_CAPACITY
        self.id_to_trip = {}      # request id
        self.destination_to_id = {}  #destination nodes
        self.origin_to_id = {}  #pickup nodes
        self.id_to_wait_start={}    #passenger_wait_start_time
        self.id_to_pickup={}
        self.id_to_dropoff={}
        self.next_node = None
        self.agent_process = self.env.process(self.run())

    def track_idle(self):
        idle_time = self.env.now - self.idle_start
        self.total_idle += idle_time
        log_event(self.env.now, 'AV_IDLE', av_i=self.id_num, node=self.current_node, period=idle_time)
        self.idle_start = self.env.now
        self.stat_collector.av_idle_time_queue.append(idle_time)

    def run(self):
        while True:
            self.idle_start = self.env.now
            self.trip_assigned = False
            self.current_destination = None
            self.current_route = None
            self.is_idle = True
            try:
                yield self.env.timeout(60*25)
            except simpy.Interrupt:
                #trip = self.trip_queue.popleft()
                # idle_time = self.env.now - self.idle_start
                # self.total_idle += idle_time
                # log_event(self.env.now, 'AV_IDLE', av_i=self.id_num, node=self.current_node, period=idle_time)
                # if self.current_route is None: #just knocked to update idle time, no trips yet
                #     continue
                self.track_idle()
                self.is_idle = False
                #trip = self.id_to_trip[self.next_id_to_fill - 1]
                #log_event(self.env.now, 'AV_IDLE', av_i=self.id_num, trip_i=trip.trip_id, node=self.current_node, period=idle_time)

                self.trip_assigned = True
                #self.current_destination = trip.destination
                #self.current_route = deque()#trip.route
                while True:
                    #this_node = self.current_node
                    if self.current_node in self.origin_to_id:
                        yield self.env.process(self.pickup())
                    if self.current_route[0] == self.current_node: #expected
                        self.current_route.popleft()
                        self.next_node = self.current_route.popleft()
                    elif self.current_route[0] == self.next_node:
                        #next_node = self.current_route.popleft()
                        self.current_route.popleft()
                        # l = get_link_length(self.current_node,  self.next_node)
                    # else: #bad case
                    #     self.current_node = self.current_route.popleft()
                    #     self.next_node = self.current_route.popleft()
                        #l = get_link_length(self.current_node, self.current_route[0])
                    #this_node = self.current_node
                    #next_node = self.next_node
                    next_next_node = self.current_route[0] if len(self.current_route) > 0 else None
                    yield self.env.process(self.traverse_link(self.current_node, self.next_node, next_next_node))

                    if self.current_node in self.destination_to_id:
                        yield self.env.process(self.dropoff())
                    if len(self.id_to_trip) == 0 or len(self.current_route) == 0 or self.next_node is None:
                        break

    def pickup(self):
        seats_for_pickup = self.origin_to_id.pop(self.current_node)
        for seat in seats_for_pickup:
            trip = self.id_to_trip[seat]
            trip.pickup_time = self.env.now
            self.id_to_pickup[seat]=self.env.now
            yield trip.indiv_travel_queue.put(TripUpdate(self.current_node, None, None, 'start', self.id_num))
            log_event(self.env.now, 'AV_PICKUP_INDIV', indiv_i=trip.individual, av_i=self.id_num, trip_i=trip.trip_id, node=self.current_node)

    def dropoff(self):
        global completed_trip
        for ID in self.destination_to_id[self.current_node]:
            #if ID in self.id_to_trip:
            if self.id_to_trip[ID].origin in self.origin_to_id:
                return
        ids_to_drop = self.destination_to_id.pop(self.current_node)
        for ID in ids_to_drop:
            trip = self.id_to_trip.pop(ID)
            for companion_id in range(1, trip.num_people):
                self.id_to_trip.pop(ID + companion_id)
            #self.id_to_trip[id] = None#cleanup
            #self.next_seat_to_fill -= trip.num_people #cleanup
            yield trip.indiv_travel_queue.put(TripUpdate(self.current_node, None, None, 'end', self.id_num))
            completed_trip = completed_trip +1
            trip_time = self.env.now - trip.pickup_time
            print("Trip complete", completed_trip)
            log_event(self.env.now, 'AV_DROP_INDIV', indiv_i=trip.individual, av_i=self.id_num, trip_i=trip.trip_id, node=self.current_node, period=trip_time)
            #todo: stat_collector

    def compute_current_idle_time(self, now):
        return (now - self.idle_start) * self.is_idle

    def assign_trip(self, trip: TripRequest, route: deque):

        if trip.destination in self.destination_to_id.keys():
            self.destination_to_id[trip.destination].append(self.next_id_to_fill)
        else:
            self.destination_to_id[trip.destination] = [self.next_id_to_fill]

        if trip.origin in self.origin_to_id.keys():
            self.origin_to_id[trip.origin].append(self.next_id_to_fill)
        else:
            self.origin_to_id[trip.origin] = [self.next_id_to_fill]

        for _ in range(trip.num_people):
            self.id_to_trip[self.next_id_to_fill] = trip
            self.id_to_wait_start[self.next_id_to_fill]=trip.time
            self.next_id_to_fill += 1
        self.current_route = route

    def traverse_link(self, start, end, next_next_node):#, indiv_travel_queue: simpy.Store, state: str):
        link_capacity = GLOBAL.LINK_CAPACITY[(start, end)]
        length = get_link_length(start, end)
        travel_time = 1.0 * length / GLOBAL.CAR_SPEED_METER_MINUTES
        # if state == 'end':
        #     travel_time += GLOBAL.TRIP_END_OVERHEAD_MINUTES
        # x, y = get_x_y(end)
        next_req = None
        with link_capacity.request() as req:
            req_time = self.env.now
            # Request one space in the link
            yield req
            # Actual link traverse time
            yield self.env.timeout(travel_time)
            # Before releasing this link, request one space in the next link
            if next_next_node is not None:
                next_link_capacity = GLOBAL.LINK_CAPACITY[(end, next_next_node)]
                next_req = next_link_capacity.request()
                yield next_req
            congestion = self.env.now - req_time - travel_time
            self.stat_collector.congestion_time_queue.append(congestion)
        if next_req is not None:
            next_link_capacity.release(next_req)
        if len(self.id_to_trip) == 1 and len(self.origin_to_id) > 0:
            self.stat_collector.av_empty_time_queue.append(travel_time + congestion)
            log_event(self.env.now, 'AV_EMPTY', av_i=self.id_num, node=self.current_node, period=travel_time + congestion)
        #self.env.timeout(travel_time)
        # if indiv_travel_queue is not None:
        #     indiv_travel_queue.put(TripUpdate(end, x, y, state))
        self.current_node = end
        self.next_node = next_next_node
        #if state == 'progress' or state == 'end':
        self.total_effective_distance += length
        log_event(self.env.now, 'AV_TRAVEL_LINK', av_i=self.id_num, node=self.current_node, period=length, num_people=len(self.id_to_trip))

    # def traverse_route(self, origin, destination, route, indiv_travel_queue: simpy.Store):
    #     this_node = origin
    #     for node_i,next_node in enumerate(route):
    #         if next_node == origin:
    #             continue
    #         state = 'end' if next_node == destination else 'progress'
    #         next_next_node = route[node_i+1] if node_i+1 < len(route) else None
    #         #self.traverse_link(self.current_node, next_node, indiv_travel_queue, state)
    #         traverse_link = self.env.process(self.traverse_link(this_node, next_node, next_next_node, indiv_travel_queue, state))
    #         try:
    #             yield traverse_link
    #         except simpy.Interrupt:
    #             yield traverse_link
    #         this_node = next_node
    #
    #
    # def start_trip(self, origin, destination, route, indiv_travel_queue: simpy.Store, indiv, trip_id):
    #     #logging.info('AV-%d pickedup Individual-%d trip-%d at %s' % (self.id_num, indiv, trip_id, str(datetime.timedelta(minutes=self.env.now)) ))
    #     log_event(self.env.now, 'AV_PICKUP_INDIV', indiv_i=indiv, av_i=self.id_num, trip_i=trip_id, node=self.current_node)
    #     yield indiv_travel_queue.put(TripUpdate(None, None, None, 'start'))
    #     self.trip_started = True
    #     #self.last_trip_start_time = self.env.now
    #     trip_start_time = self.env.now
    #     if route is None:
    #         route = nx_shortest_path(source=origin, target=destination, weight='length')
    #         #self.current_route = route
    #     logging.debug(route)
    #     self.traverse_process = self.env.process(self.traverse_route(origin, destination, route, indiv_travel_queue))
    #     yield self.traverse_process
    #     #logging.info('AV-%d dropped Individual-%d trip-%d on node-%d at %s' % (self.id_num, indiv, trip_id, destination, str(datetime.timedelta(minutes=self.env.now)) ))
    #     trip_end_time = self.env.now
    #     log_event(self.env.now, 'AV_DROP_INDIV', indiv_i=indiv, av_i=self.id_num, trip_i=trip_id, node=self.current_node, period=trip_end_time-trip_start_time)
    #     self.trip_started = False
    #
    # def fetch_indiv(self, destination, route, indiv_travel_queue: simpy.Store, indiv, trip_id): #fetch indiv by updating node by node
    #     #logging.info('AV-%d (node-%d) is fetching Individual-%d (nide-%d) at %s' % (self.id_num, self.current_node, indiv, destination,
    #     #                                                                            str(datetime.timedelta(minutes=self.env.now)) ))
    #     log_event(self.env.now, 'AV_FETCH_INDIV', indiv_i=indiv, av_i=self.id_num, trip_i=trip_id, node=self.current_node)
    #     self.trip_started = False
    #     self.empty_trip_start = self.env.now
    #
    #     if route is None:
    #         route = nx_shortest_path(source=self.current_node, target=destination, weight='length')
    #         #self.current_route = route
    #     #track wasteage of fuel by saving starttime of empty trip
    #     logging.debug(route)
    #     start = self.current_node
    #     self.traverse_process = self.env.process(self.traverse_route(start, destination, route, None))
    #     yield self.traverse_process
    #     self.idle_start = self.env.now
    #     self.empty_trip_end = self.env.now
    #     empty_time = self.empty_trip_end - self.empty_trip_start
    #     self.total_empty_trip += empty_time
    #     self.stat_collector.av_empty_time_queue.append(empty_time)
    #     #logging.info('AV-%d reached node-%d at %s with empty duration %f' % (self.id_num, destination,
    #     #                                                               str(datetime.timedelta(minutes=self.env.now)), self.empty_trip_end - self.empty_trip_start ))
    #     log_event(self.env.now, 'AV_EMPTY', indiv_i=indiv, av_i=self.id_num, trip_i=trip_id, node=self.current_node, period=empty_time)