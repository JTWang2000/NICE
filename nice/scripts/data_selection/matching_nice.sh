#!/bin/bash

gradient_path=$1
train_file_names=$2
ckpts=$3
checkpoint_weights=$4

validation_gradient_path=$5
target_task_names=$6
output_path=$7

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

echo $gradient_path, $train_file_names, $ckpts, $validation_gradient_path, $output_path
python3 -m nice.data_selection.matching_nice \
--gradient_path $gradient_path \
--train_file_names $train_file_names \
--ckpts $ckpts \
--checkpoint_weights $checkpoint_weights \
--validation_gradient_path $validation_gradient_path \
--target_task_names $target_task_names \
--output_path $output_path