#!/bin/bash

# for validation data, we should always get gradients with sgd

task=$1 # alpaca, tldr, hh_rlhf, codex
data_dir=$2 # path to data
model=$3 # path to model
output_path=$4 # path to output
gradient_type=$5
mc=$6
temp=${7:--1} # Default temperature to -1 if not provided

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi


use_vllm=false
guidance=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --use_vllm) use_vllm=true ;;
        --use_gpt) use_gpt=true ;;
        --guidance) guidance=true ;;
    esac
    shift
done


python_command="python3 -m nice.data_selection.get_info \
--task $task \
--info_type generations \
--model_path $model \
--output_path $output_path \
--gradient_type $gradient_type \
--data_dir $data_dir \
--mc $mc \
--input_temperature $temp"

if [ "$use_vllm" = true ]; then
    python_command+=" --use_vllm"
fi
if [ "$use_gpt" = true ]; then
    python_command+=" --use_gpt"
fi
if [ "$guidance" = true ]; then
    python_command+=" --guidance"
fi

echo $python_command
# Execute the dynamically constructed Python command
$python_command