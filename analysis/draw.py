import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from operator import add
from pathlib import Path

plt.rcParams["font.family"] = "Times New Roman"

sns.set_context("talk")

total_trips = 10005
cols = ['INFO', 'root', 'TIME', 'MSG', 'INDIV', 'AV', 'TRIP', 'NODE', 'OTHER', 'NOW', 'PEOPLE']
input_data_dir = '../input/june08/basicTSP/' #5obj-roulette/'
save_dir = '../figure/basicTSP-final/'
av_list = list(range(200, 501, 50))
seat_list = [4, 7, 10]
num_run = 3
arrival_delay_slack = 0
Path(save_dir).mkdir(parents=True, exist_ok=True)
method = 'GREEDY-INDIV'
stat = 'Avg. indiv. wait'#'Early reached%'#'Avg. v. idle time'#'Avg. indiv. wait'#'Trips%'
stat_list = ['VKT reduction%', 'Avg. v. emp. time', 'Avg. v. idle time', 'Avg. indiv. wait', 'Trips%', 'Avg. indiv. a.delay']
filename = {'Trips%': 'completed-trips', 'Avg. indiv. wait': 'avg-indiv-wait', 'Avg. indiv. a.delay': 'avg-indiv-delay',
            'Avg. v. emp. time': 'avg-emp-time', 'Avg. v. idle time': 'avg-idle-time',
            'VKT reduction%': 'vkt-reduction', 'Early reached%':'early-reached'}
stat_col = []
run_col = []
av_col = []
seat_col = []


def compute_stat(stat, df_log):
    if stat == 'Trips%':
        return 100*df_log[df_log['MSG'] == 'AV_DROP_INDIV']['TRIP'].count() / total_trips
    if stat == 'Avg. indiv. wait':
        return df_log[df_log['MSG'] == 'INDIV_WAIT']['OTHER'].mean()
    if stat == 'Avg. v. emp. time':
        return df_log[df_log['MSG'] == 'AV_EMPTY']['OTHER'].mean()
    if stat == 'Avg. v. idle time':
        dfx = df_log[df_log['MSG'] == 'AV_IDLE'].groupby('AV', as_index=False).agg(IDLE=("OTHER", "sum"))
        return dfx['IDLE'].mean()#df_log[df_log['MSG'] == 'AV_IDLE']['OTHER'].mean()
    if stat == 'Avg. indiv. a.delay2':
        dfx = df_log[df_log['MSG'] == 'INDIV_ARRIVAL_DELAY']['OTHER'].copy()
        dfx[dfx < 0] = 0
        return dfx.mean()
    if stat == 'Avg. indiv. a.delay':
        dfx = df_log[df_log['MSG'] == 'INDIV_ARRIVAL_DELAY']
        dfr = pd.read_csv('../sample0-nodes-ext.csv', usecols=['trip_id', 'destination_now'])
        dfm = dfx.merge(dfr, left_on='TRIP', right_on='trip_id')
        delay = dfm['NOW'] - dfm['destination_now'] - arrival_delay_slack
        delay[delay < 0] = 0
        return delay.mean()
    if stat == 'Early reached%':
        dfx = df_log[df_log['MSG'] == 'INDIV_ARRIVAL_DELAY']
        dfr = pd.read_csv('../sample0-nodes-ext.csv', usecols=['trip_id', 'destination_now'])
        dfm = dfx.merge(dfr, left_on='TRIP', right_on='trip_id')
        delay = dfm['NOW'] < dfm['destination_now']
        return 100*delay.sum()/total_trips
    if stat == 'VKT reduction%':
        dfx = df_log[df_log['MSG'] == 'AV_TRAVEL_LINK']
        dfx['PEOPLE'] = dfx['PEOPLE'].astype('int')
        dfx['vkt'] = dfx['OTHER'] / dfx['PEOPLE']
        n_vkt = dfx['vkt'].sum()
        dfr = pd.read_csv('../sample0-nodes-ext.csv', usecols=['trip_id', 'destination_distance'])
        dfy = df_log[df_log['MSG'] == 'AV_DROP_INDIV']
        dfm = dfy.merge(dfr, left_on='TRIP', right_on='trip_id')
        o_vkt = dfm['destination_distance'].sum()
        return 100*(n_vkt-o_vkt)/o_vkt


#for stat in stat_list:
for av_i, av in enumerate(av_list):
    for seat in seat_list:
        df = pd.read_csv(input_data_dir + method + '-run-0-av-' + str(av) + '-s-' + str(seat) + '.log', delimiter=';', names=cols, skiprows=1) #log/rnd-weight/ rnd-weight run-0 av-300.csv
        sep_id = df[df['INDIV'] == 'INDIV'].index.to_list()
        separator = [-1]
        separator.extend(sep_id)
        separator.append(len(df))
        actual_num_run = num_run
        if len(separator) <= num_run:
            actual_num_run = len(separator)-1
            print('!!WARNING!!! %s av-%d, seat-%d has (%d) less runs than %d runs' % (method, av, seat, actual_num_run, num_run))
        for run_i in range(actual_num_run):
            df_log = df.iloc[separator[run_i]+1:separator[run_i+1], :].copy()
            df_log['OTHER'] = df_log['OTHER'].astype('float64')
            run_col.append(run_i)
            seat_col.append(seat)
            av_col.append(av)
            stat_col.append(compute_stat(stat, df_log))
df_result = pd.DataFrame()
df_result['AV'] = av_col
df_result['SEAT'] = seat_col
df_result['RUN'] = run_col
df_result[stat] = stat_col

g = sns.catplot(x="AV", y=stat,
                hue="SEAT",data=df_result, kind="bar", ci=None, legend=False)
if stat != 'VKT reduction%':
    g.ax.set_yscale('log')
plt.legend(loc='upper center',  bbox_to_anchor=(0.50, 1.12),
           ncol = 3, prop={'size': 10}, frameon=False)
#plt.show()

# g=sns.displot(data=df_result, x=stat, kind="ecdf", hue='AV', col='SEAT', palette=sns.color_palette("icefire", as_cmap=True))
# g.set(ylabel='Arrival delay')
# g.ax.set_xscale('log')
#
# plt.grid(True, which="both", ls="dotted", c='gray')
plt.savefig(save_dir + filename[stat] + '.pdf', bbox_inches='tight')
plt.clf()





