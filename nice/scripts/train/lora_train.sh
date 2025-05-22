#!/bin/bash

source nice/scripts/train/base_training_args.sh

train_files=$1
model_path=$2
job_name=$3
data_seed=$4

output_dir=/home/NICE/out/${job_name}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

training_args="$base_training_args \
--model_name_or_path $model_path \
--output_dir $output_dir \
--data_seed $data_seed \
--train_files ${train_files[@]} 2>&1 | tee $output_dir/train.log"

echo "$header $training_args"
eval "$header" "$training_args"