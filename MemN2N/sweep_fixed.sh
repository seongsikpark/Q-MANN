#!/bin/bash
make clean
make

for ((i=0;i<2;i++)) do
    ./MemN2N 2 1 20 $i
    mv result.csv result_att_2_${i}.csv
done
