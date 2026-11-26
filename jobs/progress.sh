#!/bin/bash
# usage: ./progress.sh JOBID   (refreshes every 15 s; Ctrl+C to exit, the job keeps running)
J=${1:?usage: ./progress.sh JOBID}; L=logs/train_$J.log
while true; do
  clear
  Q=$(squeue -h -j $J -o '%T | elapsed %M | node %N' 2>/dev/null)
  echo "Job $J | ${Q:-finished or cancelled (not in queue)}"
  n=$(grep -cE '^ep[0-9]+' $L)
  t=$(grep -E '^ep[0-9]+' $L | tail -1 | grep -oE '[0-9]+s' | tail -1 | tr -d s)
  echo "Epochs done: $n/40 | last epoch: ${t:-?} s | max time left: ~$(( ${t:-0} * (40 - n) / 60 )) min (early stop may finish sooner)"
  echo; echo "--- last epochs ---";            grep -E '^ep[0-9]+' $L | tail -8
  echo; echo "--- best checkpoint so far ---"; grep -E '^ep[0-9]+.*\*$' $L | tail -1
  echo; echo "--- log tail ---";               tail -3 $L
  [ -z "$Q" ] && break
  sleep 15
done
