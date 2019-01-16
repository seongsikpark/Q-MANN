#!/bin/bash
make clean
make


for ((iwl=5;iwl<=5;iwl++)) do
#for ((iwl=0;iwl<=0;iwl++)) do

    #name_result_file=result_170410_1_${iwl}_1.csv
    name_result_file=result_170727.csv
    idx_start=1
    idx_end=20
    
    rm $name_result_file
    touch $name_result_file
    
    for ((i=$idx_start;i<=$idx_end;i++)) do
        ./MemN2N 10 $i $i $iwl
        #./MemN2N 1 $i $i $iwl
    
        if [ $i -eq $idx_start ] 
        then
            cat result_all.csv >> $name_result_file
        else
            cat result.csv >> $name_result_file
        fi
    done

done
