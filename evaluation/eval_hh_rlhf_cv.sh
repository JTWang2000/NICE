#!/bin/bash
source eval.sh
# Directory containing the checkpoints
CHECKPOINT_DIR=$1  # Directory where checkpoints are stored
MODEL=$2           # Model name (e.g., llama)
CUDA=$3

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=$CUDA

# Temporary files to store results
TEMP_VAL_FILE="$CHECKPOINT_DIR/rlhf_val_results.txt"
echo "Validation results stored in: $TEMP_VAL_FILE"

# main RLHF evaluation function
eval_rlhf() {
    mdir=$1
    model=$2
    split=$3
    set_save_dir $mdir "hh_rlhf_$split"
    mkdir -p $save_dir

    if [[ "$model" == "llama" ]]; then
        chat_formatting_function="eval.templates.create_hhrlhf_prompt_with_tulu_chat_format"
    elif [[ "$model" == "mistral" ]]; then
        chat_formatting_function="eval.templates.create_hhrlhf_prompt_with_llama2_chat_format"
    elif [[ "$model" == "llama3" ]]; then
        chat_formatting_function="eval.templates.create_hhrlhf_prompt_with_llama3_chat_format"
    else
        echo "Unsupported model type"
        return 1
    fi

    if [[ "$split" == "test" ]]; then
        data_file="$DATA_DIR/hh_rlhf/test/hh_rlhf_test_data.jsonl"
    elif [[ "$split" == "eval" ]]; then
        data_file="$DATA_DIR/hh_rlhf/eval/hh_rlhf_validation_data.jsonl"
    fi

    cmd="python -m eval.hh_rlhf.run_eval \
    --data_file $data_file \
    --max_context_length 512 \
    --save_dir $save_dir \
    --model $mdir \
    --tokenizer $mdir \
    --eval_batch_size 20 \
    --use_chat_format \
    --convert_to_bf16 \
    --chat_formatting_function $chat_formatting_function"
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

# Define the function to extract evaluation results
extract_rlhf() {
    mdir=$1
    split=$2
    set_save_dir $mdir "hh_rlhf_$split"
    # Extract the first floating point number from the file, which represents the F1 score
    result=$(grep -oP '^-?\d+\.\d+' $save_dir/metrics.json | head -n 1)
    echo $result
}

# Check if val_results.txt already exists
if  [ -f "$TEMP_VAL_FILE" ]; then
    echo "Validation results file exists, finding the best checkpoint directly."
    # Find the checkpoint with the highest total score
    best_checkpoint=$(sort -k2 -n -r $TEMP_VAL_FILE | head -n 1 | awk '{print $1}')
    echo "Best checkpoint from existing validation results: $best_checkpoint"
else
    echo "Validation results file does not exist, running evaluation on checkpoints."

    # Iterate through all checkpoints in the directory with the pattern 'checkpoint-*'
    for checkpoint in "$CHECKPOINT_DIR"/checkpoint-*; do
        echo "Evaluating checkpoint on validation set: $checkpoint"
        eval_rlhf $checkpoint $MODEL eval

        # Extract the evaluation performance based on F1 score
        f1_score=$(extract_rlhf $checkpoint eval)

        # Save checkpoint and its F1 score for validation
        echo "$checkpoint $f1_score" >> $TEMP_VAL_FILE
    done

    # Find the checkpoint with the highest F1 score for validation
    best_checkpoint=$(sort -k2 -n -r $TEMP_VAL_FILE | head -n 1 | awk '{print $1}')
    echo "Best checkpoint after validation evaluation: $best_checkpoint"
fi

eval_rlhf $best_checkpoint $MODEL test

