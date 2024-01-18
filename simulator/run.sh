#!/bin/bash
cd "$(dirname "$0")"

balance='INDIV' #'INDIV', 'AV', 'BOTH', 'ADAPT', 'RND', 'ADAPT_OLD' ADAPT_OLD_MIRROR
run=10
del=50 #25
start=300 #300
end=900 #900

for ((i=start; i<=end; i=i+del))
do
  #echo "Start running simulation for av=$i"
  python run_experiment.py -av=$i -balance=$balance -run=$run &
  pids[${i}]=$!
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

