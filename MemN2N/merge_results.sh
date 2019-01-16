#!/bin/bash

dir_target=results_f_sweep_emb200

echo "target_dir: "${dir_target}

rm ./${dir_target}/result_sw_f_total.csv
touch ./${dir_target}/result_sw_f_total.csv

for ((i=1;i<6;i++)) do
    for ((j=1;j<10;j++)) do
        cat ./${dir_target}/result_sw_${i}_${j}.csv >> ./${dir_target}/result_sw_f_total.csv
    done
done
