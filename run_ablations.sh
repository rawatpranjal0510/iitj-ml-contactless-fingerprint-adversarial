#!/bin/bash
cd /iitjhome/m24ma2008/Pranjal/MLProject/Codes; PY=~/.conda/envs/fp_env/bin/python
$PY policy.py                                       > logs/policy.log            2>&1   # re-run main (adds params JSON; same results)
$PY policy.py --metric mclfiq                       > logs/policy_mclfiq.log     2>&1
$PY policy.py --primary 0.02 --suffix _eps0.02      > logs/policy_eps0.02.log    2>&1
$PY policy.py --far_budget 0.0012 --suffix _far0.12 > logs/policy_far0.12.log    2>&1
$PY policy.py --far_budget 0.002  --suffix _far0.20 > logs/policy_far0.20.log    2>&1
echo "ABLATIONS DONE $(date)" > logs/ablations.done
