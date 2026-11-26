#!/bin/bash
# Rebuild results/ and paper/ from out/ (re-run after any re-analysis). Safe to run repeatedly.
set -e; cd /iitjhome/m24ma2008/Pranjal/MLProject/Codes; R=results
rm -rf $R paper; mkdir -p $R/tables $R/figures $R/runs $R/logs $R/data paper/figures
cp out/tables/*.csv out/tables/*.tex out/tables/*.json $R/tables/; rm -f $R/tables/*limit64*
cp out/figs/paper/*.png out/figs/paper/*.pdf $R/figures/
cp out/tau.json out/quality_cutoffs.json out/split.json $R/
cp out/runs/* $R/runs/
cp out/images.csv out/quality.csv out/attacks_test.csv out/attacks_val.csv out/attacks_test_strong.csv $R/data/
cp logs/analyze_test.log logs/analyze_val.log logs/policy*.log $R/logs/
cp logs/attack_600808.log $R/logs/attack_test.log; cp logs/attack_600809.log $R/logs/attack_val.log
cp logs/attack_601182.log $R/logs/attack_test_strong.log
unzip -oq paper_src.zip -d paper; cp out/figs/paper/*.pdf paper/figures/
echo "results/: $(find $R -type f | wc -l) files, $(du -sh $R | cut -f1) | paper/: $(find paper -type f | wc -l) files"
