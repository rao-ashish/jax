#!/bin/bash
# A launcher script that spawns 4 processes running the specified script
# (defaults to mlp_forward_pass.py), each managing two devices.

script=${1:-mlp_forward_pass.py}
num_processes=4
devices_per_process=2

range=$(seq 0 $(($num_processes - 1)))

for i in $range; do
  python $script $i $num_processes $devices_per_process > /tmp/toy_$i.out &
done

wait

for i in $range; do
  echo "=================== process $i output ==================="
  cat /tmp/toy_$i.out
  echo
done