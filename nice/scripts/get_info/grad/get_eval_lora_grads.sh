#!/bin/bash

task=$1 # alpaca, tldr, hh_rlhf, codex
data_dir=$2 # path to data
model=$3 # path to model
output_path=$4 # path to output
gradient_type=$5
policy=$6 # [vanilla | rejection | hard_rejection | topk_rejection | topk_hard]; default is always vanilla

# Check if use_cache_generation (7th argument) is provided
if [ -z "$7" ]; then
    use_cache_generation=-1
else
    use_cache_generation=$7
fi

worst_case_protect=false
guidance=false
do_normalize=false
use_vllm=false
use_gpt=false

# Parse optional named flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --worst_case_protect) worst_case_protect=true ;;
        --guidance) guidance=true ;;
        --do_normalize) do_normalize=true ;;
        --use_vllm) use_vllm=true ;;
        --use_gpt) use_gpt=true ;;
    esac
    shift
done

#if [[ ! -d $output_path ]]; then
#    mkdir -p $output_path
#fi

echo $use_vllm
python_command="python3 -m nice.data_selection.get_info \
--task $task \
--info_type grads \
--model_path $model \
--output_path $output_path \
--gradient_type $gradient_type \
--data_dir $data_dir \
--policy $policy \
--use_cache_generation $use_cache_generation "

# Add optional flags only if true
if [ "$worst_case_protect" = true ]; then
    python_command+=" --worst_case_protect"
fi
if [ "$guidance" = true ]; then
    python_command+=" --guidance"
fi
if [ "$do_normalize" = true ]; then
    python_command+=" --do_normalize"
fi
if [ "$use_vllm" = true ]; then
    python_command+=" --use_vllm"
fi
if [ "$use_gpt" = true ]; then
    python_command+=" --use_gpt"
fi

echo $python_command
# Execute the dynamically constructed Python command
$python_command
