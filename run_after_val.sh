#!/bin/bash
cd /iitjhome/m24ma2008/Pranjal/MLProject/Codes; PY=~/.conda/envs/fp_env/bin/python
while squeue -h -j 600809 | grep -q .; do sleep 30; done
grep -q "^End:" logs/attack_600809.log && [ -f out/attacks_val.csv ] || { echo "val attack did not finish cleanly" > logs/run_after_val.log; exit 1; }
$PY analyze.py --file attacks_val  > logs/analyze_val.log  2>&1
$PY analyze.py --file attacks_test > logs/analyze_test.log 2>&1
$PY policy.py                      > logs/policy.log       2>&1
echo "ALL DONE $(date)" > logs/run_after_val.log
