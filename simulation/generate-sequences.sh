#!/bin/bash
source ~/.profile

l=100 #length of sequences
s=100 #number of pairs for each d
d=35 #exact s pairs for <= d. For d < ED <= l, there are fewer pairs. 

dir=seq-n$l-ED$d #output dir
mkdir -p $dir
cd $dir

for r in {1..5} # r rounds, for parallel computing
do
    file=seq-n$l-ED$d-$r.txt
    rm $file
    nohup python3 ../seqSim.py --l $l --s $s --d $d --r $r > seqSim-$r.log 2>&1 &
done
