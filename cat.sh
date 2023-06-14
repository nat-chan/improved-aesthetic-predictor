#!/bin/bash
num=1000000

for i in $(seq 0 9); do
    cat $((i*num))_${num}.dat
done|grep -v 'Setting up PyTorch plugin'|sort -r > cat_0_9.dat