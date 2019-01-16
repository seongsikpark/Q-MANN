#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo "need argument #1: experiment number"
    exit 1
fi

exp_num=$1

echo "experiment number: "${exp_num}

mv softmax_input_0to24.csv softmax_input_${exp_num}_0to24.csv
mv softmax_input_25to49.csv softmax_input_${exp_num}_25to49.csv
mv softmax_input_50to74.csv softmax_input_${exp_num}_50to74.csv
mv softmax_input_75to99.csv softmax_input_${exp_num}_75to99.csv

echo "softmax input name change done"

mv softmax_output_0to24.csv softmax_output_${exp_num}_0to24.csv
mv softmax_output_25to49.csv softmax_output_${exp_num}_25to49.csv
mv softmax_output_50to74.csv softmax_output_${exp_num}_50to74.csv
mv softmax_output_75to99.csv softmax_output_${exp_num}_75to99.csv

echo "softmax output name change done"
