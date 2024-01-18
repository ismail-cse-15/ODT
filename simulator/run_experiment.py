import simpy
import numpy as np
from scipy.spatial.distance import cdist, cosine
import networkx as nx
from config import GLOBAL, get_link_length, get_x_y, get_travel_time_minutes
from mop import AssessAVsToServeTrip
from agents import TripRequest, AutonomousVehicle, Individual
import pandas as pd
import matplotlib.pyplot as plt
from statistics import StatCollector
import logging
from datetime import datetime
import argparse
from simulation import SimulatedWorld
from pathlib import Path


outpur_dir = '../june08-basicTSP'
if __name__ == '__main__':
    seeds = [24204460, 255701664, 255452145, 243420000, 233962792, 70460023, 164387811, 95284976, 248008688, 236908554,
             250416188, 60094985, 28918358, 52010314, 261135298, 14898472, 86355748, 266651191, 90120307, 166684931]
    parser = argparse.ArgumentParser(description='DRT simulation parameters')
    parser.add_argument('--av', '-av', dest='av_count', type=int, default=200,
                        help='Number of AVs (default: 200)')
    parser.add_argument('--run', '-run', dest='run_count', type=int, default=1,
                        help='Number of indep. runs (default: 1)')
    parser.add_argument('--seat', '-seat', dest='seat_count', type=int, default=4,
                        help='Number of seats (default: 4)')
    parser.add_argument('--balance','-balance', dest='balance',
                        choices=['INDIV', 'AV', 'BOTH', 'ADAPT', 'RND', 'ADAPT_OLD', 'ADAPT_OLD_MIRROR'],
                        help='How to balance between supply and demand', type=str, default='ADAPT')
    parser.add_argument('--greedy', '-greedy', dest='apply_greedy', type=int, default=0,
                        help='Apply greedy (default: 0)')
    parser.add_argument('--gtype','-gtype', dest='greedy_type',
                        choices=['INDIV', 'AV', 'BOTH', 'RND'],
                        help='How to balance between supply and demand in greedy', type=str, default='BOTH')
    args = parser.parse_args()
    GLOBAL.NUMBER_OF_AV = args.av_count
    GLOBAL.BALANCE_TYPE = args.balance
    GLOBAL.APPLY_GREEDY = False
    GLOBAL.CAR_SEAT_CAPACITY = args.seat_count
    method = args.balance
    if args.apply_greedy > 0:
        GLOBAL.APPLY_GREEDY = True
        GLOBAL.GREEDY_TYPE = args.greedy_type
        method = 'GREEDY-' + args.greedy_type        
    Path(outpur_dir).mkdir(parents=True, exist_ok=True)
    for run_i in range(0, args.run_count):
        GLOBAL.RUN_ID = run_i
        print('Start running DRT Simulator: run-%d, av-%d, balance-%s, greedy-%d-%s'%(run_i, GLOBAL.NUMBER_OF_AV, GLOBAL.BALANCE_TYPE, GLOBAL.APPLY_GREEDY, GLOBAL.GREEDY_TYPE))
        np.random.seed(seeds[run_i])
        msg = method + '-run-' + str(run_i) + '-av-' + str(GLOBAL.NUMBER_OF_AV)
        today = datetime.now()
        timestamp = ''#today.strftime("%d-%m-%y %H:%M")
        logging.basicConfig(filename=outpur_dir + '/' + timestamp + msg + '.log', level=logging.INFO, filemode='w', format='%(levelname)s;%(name)s;%(message)s')
        logging.info('TIME;MSG;INDIV;AV;TRIP;NODE;OTHER;NOW')
        env = simpy.Environment(initial_time=GLOBAL.RUN_START_HOUR*60) #initial_time=5*60
        world = SimulatedWorld(env)
        env.run(until=(GLOBAL.RUN_UNTIL_HOUR+0.001)*60)
        stat = world.stat_collector.stat_df
        #plt.plot(stat['minutes'], stat['indiv-wait'])
        #plt.plot(stat['minutes'], stat['indiv-wait'])
        stat.plot.line('minutes', ['indiv-wait', 'av-empty', 'av-idle']) #, 'av-idle'
        plt.yscale('log')
        #plt.savefig(outpur_dir + '/' + timestamp  + msg + '.png', format='png', bbox_inches='tight')
        stat.to_csv(outpur_dir + '/' + timestamp  + msg + '.csv', index=False)
        # plt.show()
        #print(stat_df.head())