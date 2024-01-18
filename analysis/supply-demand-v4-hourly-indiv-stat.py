import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
sns.set_context("talk")
plt.rcParams["font.family"] = "Times New Roman"

total_trips = 10005
cols = ['INFO', 'root', 'TIME', 'MSG', 'INDIV', 'AV', 'TRIP', 'NODE', 'OTHER', 'NOW', 'PEOPLE']
input_data_dir = '../input/june08/basicTSP/' #5obj-roulette/'
save_dir = '../figure/basicTSP-final/'
av_list = list(range(400, 401, 50))
seat_list = [4, 7, 10]
seat = 4
num_run = 3
arrival_delay_slack = 0
Path(save_dir).mkdir(parents=True, exist_ok=True)
method = 'GREEDY-INDIV'
minutes_del = 60
# = (27+1 - 3)/(minutes_del/60.0)
hour_list = np.arange(3*60, 25*60, minutes_del)
hour_list2 = np.arange(3, 25, 1)

#['INDIV', 'GREEDY-RND', 'RND' ]#['INDIV', 'ADAPT_OLD', 'BOTH', 'GREEDY-BOTH', 'GREEDY-INDIV', 'RND']#['ADAPT_OLD', 'ADAPT_OLD_EUCLID', 'GREEDY-INDIV', 'RND']
    ##['INDIV', 'AV', 'BOTH', 'ADAPT', 'RND', 'ADAPT_OLD', 'ADAPT_OLD_MIRROR', 'GREEDY-INDIV', 'GREEDY-AV', 'GREEDY-BOTH']
Path(save_dir).mkdir(parents=True, exist_ok=True)
for av_i, av in enumerate(av_list):
    idle = np.zeros((len(hour_list), num_run))
    req = np.zeros((len(hour_list), num_run))
    pick = np.zeros((len(hour_list), num_run))
    drop =  np.zeros((len(hour_list), num_run))
    df = pd.read_csv(input_data_dir + method + '-run-0-av-' + str(av) + '-s-' + str(seat) + '.log', delimiter=';', names=cols, skiprows=1) #log/rnd-weight/ rnd-weight run-0 av-300.csv
    sep_id = df[df['INDIV'] == 'INDIV'].index.to_list()
    separator = [-1]
    separator.extend(sep_id)
    separator.append(len(df))
    actual_num_run = num_run
    if len(separator) <= num_run:
        actual_num_run = len(separator)-1
        print('!!WARNING!!! %s av-%d has (%d) less runs than %d runs' % (method, av, actual_num_run, num_run))
    for run_i in range(actual_num_run):
        df_log = df.iloc[separator[run_i]+1:separator[run_i+1], :].copy()#pd.read_csv(input_data_dir + method + '-run-' + str(run_i) + '-av-' + str(av) + '.csv', delimiter=',', nrows=288) #log/rnd-weight/ rnd-weight run-0 av-300.csv
        df_log['OTHER'] = df_log['OTHER'].astype('float64')
        df_log['NOW'] = df_log['NOW'].astype('float64')
        df_log['PEOPLE'] = df_log['PEOPLE'].astype('int')
        df_log['hour'] = np.digitize(df_log['NOW'].to_numpy(), hour_list)
        #df_log['hour'] = df_log['hour'].astype('int')


        df_pick = df_log[df_log['MSG'] == 'INDIV_WAIT'].groupby(['hour'], as_index=False).agg(REQ=("OTHER", "mean"))
        #df_pick['REQ'] =  df_pick['REQ']/ total_trips
        #df_pick['REQ'] = df_pick['REQ'].cumsum()
        A = df_pick['REQ'].to_numpy()
        A = np.pad(A, (0, len(hour_list) - len(A)), 'constant')#
        pick[:, run_i] = A#/av #(av*seat)

        df_drop = df_log[df_log['MSG'] == 'INDIV_ARRIVAL_DELAY'].groupby(['hour'], as_index=False).agg(REQ=("OTHER", "mean"))
        #df_drop['REQ'] =  df_drop['REQ']/ total_trips
        #df_drop['REQ'] = df_drop['REQ'].cumsum()
        A = df_drop['REQ'].to_numpy()
        A = np.pad(A, (0, len(hour_list) - len(A)), 'constant')#
        drop[:, run_i] = A#/av #(av*seat)

        df_idle = df_log[df_log['MSG'] == 'AV_IDLE'].groupby(['hour', 'AV'], as_index=False).agg(REQ=("OTHER", "sum"))
        df_idle = df_idle.groupby(['hour'], as_index=False).agg(REQ=("REQ", "mean"))
        #df_pick['REQ'] =  df_pick['REQ']/ total_trips
        #df_pick['REQ'] = df_pick['REQ'].cumsum()
        A = df_idle['REQ'].to_numpy()
        A = np.pad(A, (0, len(hour_list) - len(A)), 'constant')#
        idle[:, run_i] = A#/av #(av*seat)

    pick_avg = np.mean(pick, axis=1)
    drop_avg = np.mean(drop, axis=1)
    idle_avg = np.mean(idle, axis=1)
    # supply_avg = np.mean(supply, axis=1)
    # empty_avg = np.mean(empty, axis=1)
    # idle_avg = np.mean(idle, axis=1)
    plt.figure(figsize=(10, 6), dpi=200)

    plt.suptitle(("#vehicle=%d, #seat=%d")% (av, seat))
    #plt.plot(hour_list, util_avg, label="v.util%") #linestyle='dashed'
    plt.step(hour_list2, pick_avg, label="avg. hourly indiv. wait", alpha=0.7)
    plt.step(hour_list2, drop_avg, label="avg. hourly indiv. a.delay", alpha=0.7)
    plt.step(hour_list2, idle_avg, label="avg. hourly vehicle idle", alpha=0.7)

    # plt.plot(av_list, supply_idle, label="Supply-Idle")
    # plt.plot(av_list, list( map(add, supply_empty, supply_idle) ), label="Supply-Combo")
    leg = plt.legend(loc='best')
    plt.xlabel('Hour of the day')
    plt.ylabel('Statistics')
    #plt.yscale('log')

    # plt.grid(True, which="both", ls="dotted", c='gray')
    plt.show()
    # plt.savefig(save_dir + method.replace('_','-') + '-sf-' + str(stress) + '-av-' + str(av) + '-hourly.pdf', bbox_inches='tight')
    plt.clf()
