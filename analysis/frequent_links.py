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
av=400
seat = 4
num_run = 3
arrival_delay_slack = 0
Path(save_dir).mkdir(parents=True, exist_ok=True)
method = 'GREEDY-INDIV'
edge_weight = np.zeros((2487, 2487))
df = pd.read_csv(input_data_dir + method + '-run-0-av-' + str(av) + '-s-' + str(seat) + '.log', delimiter=';', names=cols, skiprows=1) #log/rnd-weight/ rnd-weight run-0 av-300.csv
for av_i in range(400):
    print('running for av=%d'%(av_i))
    df_av = df[df['AV'] == av_i ]
    i_node = df_av[df_av['MSG'] == 'AV_IDLE'].iloc[0]['NODE']
    node_visited = df_av[ df_av['MSG'] == 'AV_TRAVEL_LINK']['NODE'].tolist()
    for j_node in node_visited:
        edge_weight[i_node, j_node] += 1
        edge_weight[j_node, i_node] = edge_weight[i_node, j_node]
        i_node = j_node

plt.imshow(edge_weight, cmap='hot', interpolation='spline16')
plt.show()
