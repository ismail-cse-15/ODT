from pymoo.core.problem import Problem, ElementwiseProblem
import pandas as pd
import numpy as np
from pymoo.factory import get_visualization
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.population import Population
from multiprocessing.pool import ThreadPool
from agents import TripRequest, AutonomousVehicle, Individual
from config import GLOBAL, get_shortest_travel_time, get_x_y, get_shortest_travel_distance, get_travel_time, extend_by_shortest_path
import multiprocessing
from scipy.spatial.distance import cdist
from pymoo.util.function_loader import load_function
import simpy
import networkx as nx
from networkx.algorithms import approximation as approx


class AssessAVsToServeTrip(Problem):
    def __init__(self, num_avs, num_obj, num_threads, **kwargs):
        self.num_avs = num_avs
        self.num_threads = num_threads
        self.nd_sort = load_function('fast_non_dominated_sort')
        # self.av_dist_to_origin = {}
        super().__init__(n_var=1, n_obj=num_obj,
                         **kwargs)  # super().__init__(n_var=8, n_obj=3, n_constr=7, xl=2, xu=50, **kwargs)

    def set_data(self, trip: TripRequest, av_list: list, indiv: Individual):
        self.origin_x_y = list(get_x_y(trip.origin))
        self.origin_node = trip.origin
        self.destination_x_y = list(get_x_y(trip.destination))
        self.destination_node = trip.destination
        self.trip_id = trip.trip_id
        self.av_list = av_list
        self.indiv = indiv
        self.trip_requested_at = trip.time

    def calculate_av_time_to_final_destination(self, trip_queue: simpy.Store, av):
        total_distance = get_shortest_travel_time(av.current_node, av.current_route[-1])#get_travel_distance(av.current_route, av.current_route.index(av.current_node))
        final_destination = av.current_destination
        for assigned_trip in trip_queue:
            total_distance += get_shortest_travel_time(assigned_trip.route[0], assigned_trip.route[-1])#get_travel_distance(assigned_trip.route)
            final_destination = assigned_trip.destination
        return total_distance, final_destination

    def calculate_av_distance_to_final_destination(self, trip_queue: simpy.Store, av):
        total_distance = get_shortest_travel_distance(av.current_node, av.current_route[-1])#get_travel_distance(av.current_route, av.current_route.index(av.current_node))
        final_destination = av.current_destination
        for assigned_trip in trip_queue:
            total_distance += get_shortest_travel_distance(assigned_trip.route[0], assigned_trip.route[-1])#get_travel_distance(assigned_trip.route)
            final_destination = assigned_trip.destination
        return total_distance, final_destination

    def calculate_diff_dist(self, av_current_x_y, av_current_dest_x_y, av, trip_queue):
        dist_to_origin = 0
        empty_dist = 0
        effective_dist = cdist([self.origin_x_y], [self.destination_x_y], metric=GLOBAL.DISTANCE_METRIC)[0, 0]
        if av_current_dest_x_y is None:
            dist_to_origin = cdist([self.origin_x_y], [av_current_x_y], metric=GLOBAL.DISTANCE_METRIC)[0, 0]
            empty_dist = dist_to_origin
        else:
            dist_to_final_dest, final_dest = self.calculate_av_distance_to_final_destination(trip_queue, av)
            final_dest_x_y = list(get_x_y(final_dest))
            # dist_to_origin = cdist([av_current_x_y], [av_current_dest_x_y], metric=GLOBAL.DISTANCE_METRIC)[0, 0]
            empty_dist = cdist([final_dest_x_y], [self.origin_x_y], metric=GLOBAL.DISTANCE_METRIC)[0, 0]
            dist_to_origin = dist_to_final_dest + empty_dist
    
        return dist_to_origin, empty_dist, effective_dist

    def calculate_diff_time(self, av_current_node, av_current_dest_node, av, trip_queue):
        time_to_origin = 0
        empty_time = 0
        effective_dist = get_shortest_travel_time(self.origin_node, self.destination_node)#cdist([self.origin_x_y], [self.destination_x_y], metric=GLOBAL.DISTANCE_METRIC)[0, 0]
        if av_current_dest_node is None:
            time_to_origin = get_shortest_travel_time(av_current_node, self.origin_node) #cdist([self.origin_x_y], [av_current_x_y], metric=GLOBAL.DISTANCE_METRIC)[0, 0]
            empty_time = time_to_origin
        else:
            dist_to_final_dest, final_dest = self.calculate_av_time_to_final_destination(trip_queue, av)
            #final_dest_x_y = list(get_x_y(final_dest))
            # dist_to_origin = cdist([av_current_x_y], [av_current_dest_x_y], metric=GLOBAL.DISTANCE_METRIC)[0, 0]
            empty_time = get_shortest_travel_time(final_dest, self.origin_node)#cdist([final_dest_x_y], [self.origin_x_y], metric=GLOBAL.DISTANCE_METRIC)[0, 0]
            time_to_origin = dist_to_final_dest + empty_time
        return time_to_origin, empty_time, effective_dist

    def get_diff_stat(self, av: AutonomousVehicle):
        return self.indiv.total_wait_time, av.total_empty_trip, av.total_idle, av.total_effective_distance, av.compute_current_idle_time(self.trip_requested_at)
    def calculate_rider_waiting_time(self, av: AutonomousVehicle, path):
        time=0
        route= extend_by_shortest_path(path)

        for id in av.id_to_trip.keys():
            if id in av.id_to_pickup:
                print("Pickup complete")
                time = time + (av.env.now-av.id_to_wait_start[id])
                if(av.id_to_trip[id].destination not in route):
                    print("Pickup complete Destination not found")
                time = time + get_travel_time(route[0], av.id_to_trip[id].destination, route)
            else:
                print("Pickup not complete")
                time = time + (av.env.now - av.id_to_wait_start[id])
                if (av.id_to_trip[id].origin not in route):
                    print("Pickup not complete Origin not found")
                if (av.id_to_trip[id].destination not in route):
                    print("Pickup not  complete Destination not found")
                time = time + get_travel_time(route[0], av.id_to_trip[id].origin, route)
                time = time + get_travel_time(av.id_to_trip[id].origin, av.id_to_trip[id].destination, route)
        if len(av.id_to_trip)>0:
            time = time/len(av.id_to_trip)
        return time





    def _tsp(self, av: AutonomousVehicle, path_map):
        start_node = av.current_node if av.next_node is None else av.next_node
        nodes = [start_node]
        nodes.extend(av.origin_to_id.keys())

        cycle = nodes
        if len(nodes) > 2:
            GG = nx.Graph()
            for u in nodes:
                for v in nodes:
                    if u < v:
                        GG.add_edge(u, v, weight=get_shortest_travel_distance(u, v))

            cycle = approx.simulated_annealing_tsp(GG, "greedy", source=start_node, N_inner=10, max_iterations=3)
            cycle.pop()
        nodes = [cycle[-1], self.origin_node, self.destination_node]
        #nodes = [start_node, self.origin_node, self.destination_node] #todo: discard those path where destination comes before origin
        #nodes.extend(av.origin_to_seat.keys())
        nodes.extend(av.destination_to_id.keys())

        GG = nx.Graph()
        for u in nodes:
            for v in nodes:
                if u < v:
                    GG.add_edge(u, v, weight=get_shortest_travel_distance(u, v))

        cycle1 = approx.simulated_annealing_tsp(GG, "greedy", source=cycle[-1], N_inner=10, max_iterations=3)
        cycle1.pop()
        cycle1.pop(0)
        cycle.extend(cycle1)

        if cycle.index(self.destination_node) < cycle.index(self.origin_node):
            return [np.inf, np.inf]


        dist = 0

        for node_i in range(len(cycle)-1):
            dist += get_shortest_travel_distance(cycle[node_i], cycle[node_i+1])
        waiting_time_all_rider = self.calculate_rider_waiting_time(av, cycle)
        path_map[av.id_num] = cycle

        # return [dist, len(av.id_to_trip)]
        return [waiting_time_all_rider, len(av.id_to_trip)]



    def _compute_short(self, av_id, path_map):
        # av_id = av_id[0]
        av: AutonomousVehicle = self.av_list[av_id]
        return self._tsp(av, path_map)


    def _compute_short_euclidean(self, av_id):
        # av_id = av_id[0]
        av: AutonomousVehicle = self.av_list[av_id]
        trip_queue = av.trip_queue
        # prepare input data: AV origin, destination
        av_current_x_y = list(get_x_y(av.current_node))
        av_current_dest_x_y = list(get_x_y(av.current_destination)) if av.current_destination is not None else None
        dist_to_origin, av_empty_dist, av_effective_dist = self.calculate_diff_dist(av_current_x_y, av_current_dest_x_y,
                                                                                    av, trip_queue)
        indiv_total_wait, av_total_empty_trip, av_total_idle, av_total_effective_dist, av_current_idle_time = self.get_diff_stat(av)
        return dist_to_origin / indiv_total_wait, av_empty_dist / (av_total_empty_trip ), \
               1/(av_total_idle + av_current_idle_time), -1.0 * av_effective_dist / av_total_effective_dist, len(av.trip_queue)

    def greedy_select_av(self, obj_indiv_av):
        if GLOBAL.GREEDY_TYPE == 'INDIV':
            min_index = np.where(obj_indiv_av[:, 0] == obj_indiv_av[:, 0].min())[0]
        elif GLOBAL.GREEDY_TYPE == 'AV':
            min_index = np.where(obj_indiv_av[:, 1] == obj_indiv_av[:, 1].min())[0]
        elif GLOBAL.GREEDY_TYPE == 'BOTH':
            obj_sum = obj_indiv_av.sum(axis=1)
            min_index = np.where(obj_sum == obj_sum.min())[0]
        elif GLOBAL.GREEDY_TYPE == 'RND':
            min_index = list(range(len(obj_indiv_av)))
        return min_index

    def get_nd_indices_greedy(self, x):

        if len(x) == 0:
            return -1, None
        x_map = {}
        path_map = {}
        obj_indiv_av = np.zeros((len(x), 2))
        for index, x_i in enumerate(x):
            x_map[index] = x_i
            obj_indiv_av[index] = self._compute_short(x_i, path_map)

        min_index = self.greedy_select_av(obj_indiv_av)
        for index, item in enumerate(min_index):
            if obj_indiv_av[item, 0] == np.inf:
                min_index[index] = -1
        if np.sum(min_index) == -len(min_index):
            return -1, None
        av_i = x_map[min_index[np.random.randint(0, len(min_index))]]
        return av_i, path_map[av_i]


    def get_nd_indices(self, x):
        # with multiprocessing.pool.ThreadPool(self.num_threads) as p:
        #     dcs = p.map(self._compute_short, x)
        dcs = []
        for x_i in x:
            dcs.append(list(self._compute_short(x_i)))
        # list = [x for (x, y) in dcs]
        # f['F'] = np.array(dcs)
        objectives = np.array(dcs)
        nd_indices = self.nd_sort(objectives)[0]
        return nd_indices
        # print(len(dcs))

    def _evaluate(self, x, f, *args, **kwargs):
        # with multiprocessing.Pool(self.num_threads) as p:
        # dcs = p.map(self._compute_short, x)
        dcs = []
        for x_i in x:
            dcs.append(list(self._compute_short(x_i)))
        # list = [x for (x, y) in dcs]
        # print(str(np.min(list))) #+ " , " + str(np.min(array[:, 1])))
        # f['F'] = np.array(dcs)
        objectives = np.array(dcs)
        nd_indices = self.nd_sort(objectives)[0]
        return nd_indices
        # print(len(dcs))
