#!/bin/bash
cd "$(dirname "$0")"

gtype='INDIV' #'INDIV', 'AV', 'BOTH', 'RND'
run=3
del=100 #25
start=300 #300
end=700 #900
seat=4

for ((i=start; i<=end; i=i+del))
do
  #echo "Start running simulation for av=$i"
  python run_experiment.py -av=$i -run=$run -greedy=1 -gtype=$gtype -seat=$seat &
  pids[${i}]=$!
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

