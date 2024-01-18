import _pickle as pickle
import networkx as nx
import pandas as pd
import logging
import datetime
import numpy as np
import math
from collections import deque


class GLOBAL:
    SURVEY_DATA = '../sample0-nodes.csv'
    CAR_SPEED_METER_SECONDS = (40.0 * 1000.0) / (60.0 * 60.0)  # 40 km/h
    CAR_SPEED_METER_MINUTES = (40.0 * 1000.0) / 60.0  # 40 km/h
    TRIP_END_OVERHEAD_MINUTES = 1.0
    CAR_LENGTH = 4.6
    GAP_BETWEEN_TWO_CARS = 2.62
    CAR_SEAT_CAPACITY = 7
    ARRIVAL_TIME_SLACK_MINUTES = 5
    INDIV_MAX_WAIT_MIN = 30.0
    NUMBER_OF_LANE = 2
    LINK_USAGE_FACTOR = 0.0125
    NUMBER_OF_AV = 100
    RUN_ID = 0
    NUMBER_OF_TRIPS = 10006 #full:9495 sample0:10006
    AV_START_NODE = 518
    RUN_START_HOUR = 3
    RUN_UNTIL_HOUR = 27 #1 day:25
    APPLY_GREEDY = True
    BALANCE_TYPE = 'ADAPT_OLD' #'INDIV' 'AV' 'BOTH' 'ADAPT' 'RND' 'ADAPT_OLD' ADAPT_OLD_MIRROR
    GREEDY_TYPE = 'INDIV' #'INDIV' 'AV' 'BOTH' 'RND'
    NUMBER_OF_OBJECTIVES_ASSIGN_TRIP = 5
    NUMBER_OF_THREADS: int = 15
    DISTANCE_METRIC = 'euclidean'
    G = nx.read_gpickle('../emme-graph-symmetric.pickle')
    RND_DECISION_VECTOR = False
    VERBOSE = False
    NODES_LIST = list(G.nodes)
    node_dic = pickle.load(open("../emme-graph-symmetric-node-dic.pickle", "rb"))
    node_ids = pickle.load(open("../emme-graph-symmetric-node-ids.pickle", "rb"))
    #DJK_SP = pickle.load(open("../djk-shortest-path-dic.pickle", "rb"))#np.load('../djk-sp.npz')#pickle.load(open("../djk-shortest-path-dic.pickle", "rb"))
    # DJK_SP_LEN = pickle.load(open("../djk-shortest-path-length-dic.pickle", "rb"))
    LINK_CAPACITY = {}
    df_survey = pd.read_csv(SURVEY_DATA)

    # link_length_dic = pickle.load( open( "../emme-link-length.pickle", "rb" ))
    link_length_dic_id = pickle.load(open("../emme-link-length-by-id.pickle", "rb"))
    #ADJ_MAT = np.full((np.max(G.nodes)+1, np.max(G.nodes)+1), np.inf)


GLOBAL.df_survey = GLOBAL.df_survey[GLOBAL.df_survey['origin_departure_time'].notna()] #todo: should go to preprocess code

for (i, j) in GLOBAL.G.edges:
    try:
        length = GLOBAL.link_length_dic_id[(i, j)]
        GLOBAL.G.edges[i, j]['length'] = length
    except KeyError as e:
        length = GLOBAL.link_length_dic_id[(j, i)]
        GLOBAL.G.edges[i, j]['length'] = length

# for (i, j) in GLOBAL.G.edges:
#     try:
#         length = GLOBAL.link_length_dic_id[(i, j)]
#         GLOBAL.ADJ_MAT[i, j] = length
#     except KeyError as e:
#         length = GLOBAL.link_length_dic_id[(j, i)]
#         GLOBAL.ADJ_MAT[j, i] = length
def nx_shortest_path_np(source, target, weight):
    if source == target:
        return []
    if source < target:
        return nx.dijkstra_path(GLOBAL.G, source, target)
        # return GLOBAL.DJK_SP[str((source, target))]
    else:
        return nx.dijkstra_path(GLOBAL.G, source, target)
        #path = GLOBAL.DJK_SP[str((target, source))]
        return np.flip(path)#path[::-1]

def nx_shortest_path_nx(source, target, weight='length'):
    if source == target:
        return []
    return nx.shortest_path(GLOBAL.G, source, target, weight='length')

def nx_shortest_path(source, target, weight='None'):
    if source == target:
        return []
    if source < target:
        return nx.dijkstra_path(GLOBAL.G, source, target)
        # return GLOBAL.DJK_SP[(source, target)]
    else:
        path= nx.dijkstra_path(GLOBAL.G, target, source)
        # path = GLOBAL.DJK_SP[(target, source)]
        return path[::-1]

def extend_by_shortest_path(path):
    extended = deque()
    for node_i in range(len(path)-1):
        if path[node_i] == path[node_i+1]:
            continue
        extended.extend(nx_shortest_path(path[node_i], path[node_i+1]))
        if (node_i + 2) < len(path):
            extended.pop()
    return extended

def get_travel_time(source, destination, route):

    source_idx=route.index(source)
    destination_idx=route.index(destination)
    # distance=destination_idx-source_idx
    distance=0
    for i in range(len(route)):
        if (route[i] == destination):
            break
        distance = distance + get_link_length(route[i], route[i+1])

    time= distance/GLOBAL.CAR_SPEED_METER_MINUTES
    return time


def get_shortest_travel_distance(source, target):
    if source == target:
        return 0
    if source < target:
        return nx.dijkstra_path_length(GLOBAL.G,source, target)
        #return GLOBAL.DJK_SP_LEN[(source, target)]
    else:
        return nx.dijkstra_path_length(GLOBAL.G, target, source)
        #return GLOBAL.DJK_SP_LEN[(target, source)]


def calculate_link_capacity(link_length: float):
    return math.ceil(link_length * GLOBAL.LINK_USAGE_FACTOR * GLOBAL.NUMBER_OF_LANE/ (GLOBAL.CAR_LENGTH + GLOBAL.GAP_BETWEEN_TWO_CARS))


def get_shortest_travel_time(source, target):
    return get_shortest_travel_distance(source, target) / GLOBAL.CAR_SPEED_METER_MINUTES

def get_shortest_travel_time_route(route, start_node=None):
    if len(route) == 0:
        return 0
    if start_node is None:
        start_node = route[0]
    return get_shortest_travel_distance(start_node, route[-1]) / GLOBAL.CAR_SPEED_METER_MINUTES

def get_link_length_indirect(i, j):
    try:
        length = GLOBAL.link_length_dic[(GLOBAL.node_dic[i], GLOBAL.node_dic[j])]
    except KeyError as e:
        length = GLOBAL.link_length_dic[(GLOBAL.node_dic[j], GLOBAL.node_dic[i])]
    return length


def get_link_length(i, j):
    try:
        length = GLOBAL.link_length_dic_id[(i, j)]
    except KeyError as e:
        length = GLOBAL.link_length_dic_id[(j, i)]
        #logging.debug('Treating directional link (%d,%d) as birectional' % (i, j))
    return length


# def get_link_length_alternative(i, j):
#     length_meter = GLOBAL.ADJ_MAT[i, j]
#     if length_meter == np.inf:
#         length_meter = GLOBAL.ADJ_MAT[j, i]
#     if length_meter == np.inf:
#         length_meter = GLOBAL.link_length_dic_id[(j, i)]
#     return length_meter


def save_link_length_by_node():
    emme_links = nx.read_shp('/Users/ali_nayeem/Desktop/TNDP New/Network_ODMatrices_Papers/Network/emme_links.shp')
    link_length_dic = nx.get_edge_attributes(emme_links, 'LENGTH')
    nodez = nx.read_shp('/Users/ali_nayeem/Desktop/TNDP New/Network_ODMatrices_Papers/Network/emme_nodes.shp')
    pos = nx.get_node_attributes(nodez, 'ID')  # preserve ID of GIS
    link_length_dic_id = {(pos[k[0]], pos[k[1]]): v for k, v in link_length_dic.items()}
    pickle.dump(link_length_dic_id, open('../emme-link-length-by-id.pickle', 'wb'))


def get_x_y(node):
    return GLOBAL.node_dic[node][0], GLOBAL.node_dic[node][1]


def get_travel_time_minutes(route, start=0, end=None):
    if len(route) == 0:
        return 0
    if end is None:
        end = len(route) - 1
    total_time = 0.0
    for i in range(start, end):
        length = get_link_length(route[i], route[i + 1])
        total_time += (1.0 * length / GLOBAL.CAR_SPEED_METER_MINUTES)
    return total_time


def get_travel_distance(route, start=0, end=None):
    if len(route) == 0:
        return 0
    if end is None:
        end = len(route) - 1
    total_length = 0.0
    for i in range(start, end):
        length = get_link_length(route[i], route[i + 1])
        total_length += length
    return total_length


# def get_travel_time_minutes_by_node(route, start=None, end=None):
#     if start is None:
#         start = route[0]
#         end = route[len(route) - 1]
#     total_time = 0.0
#     start_i = route.index(start)
#     end_i = route.index(end)
#     for i in range(start_i, end_i):
#         length = get_link_length(route[i], route[i + 1])
#         total_time += (1.0 * length / GLOBAL.CAR_SPEED_METER_MINUTES)
#     return total_time


def log_event(now, message, indiv_i=-1, av_i=-1, trip_i=-1, node=-1, period=-1, num_people=-1):
    #logging.info('%s;%s;Indiv-%d;AV-%d;trip-%d;node-%d;%.2f' % (str(datetime.timedelta(minutes=now)), message, indiv_i, av_i, trip_i, node, period))
    logging.info('%s;%s;%d;%d;%d;%d;%.2f;%d;%d' % (str(datetime.timedelta(minutes=now)), message, indiv_i, av_i, trip_i, node, period, now, num_people))

if __name__ == '__main__':
    print("Hello")
    # path_dic = {}
    # dist_dic = {}
    # for key, value in GLOBAL.DJK_SP.items():
    #     path_dic[str(key)] = value
    # np.savez('../djk-sp.npz', **path_dic)
#     for i in GLOBAL.G.nodes:
#         for j in GLOBAL.G.nodes:
#             if i < j:
#                 try:
#                     # out = nx.bidirectional_dijkstra(GLOBAL.G, i, j, weight='length')
#                     # path_dic[(i, j)] = out[1]
#                     # dist_dic[(i, j)] = out[0]
#                     #path_dic[(j, i)] = path_dic[(i, j)]
#                     if GLOBAL.DJK_SP[(i, j)] is None:
#                         print("Not found betn %d to %d" % (i,j))
#                     #print(nx.reconstruct_path(i, j, GLOBAL.FW_PREDECESSOR))
#                     #print(nx.astar_path(GLOBAL.G, i, j, heuristic=shortest_path_distance, weight='length'))
#                 except KeyError as e:
#                     print("Not found betn %d to %d" % (i,j))
# #     pickle.dump(path_dic, open('../djk-shortest-path-dic.pickle', 'wb'))
#     pickle.dump(dist_dic, open('../djk-shortest-path-length-dic.pickle', 'wb'))




