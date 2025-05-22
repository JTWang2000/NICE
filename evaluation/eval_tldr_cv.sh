#!/bin/bash
source eval.sh
# Directory containing the checkpoints
CHECKPOINT_DIR=$1  # Directory where checkpoints are stored
MODEL=$2           # Model name (e.g., llama)
CUDA=$3

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=$CUDA

# Temporary file to store results
TEMP_FILE="$CHECKPOINT_DIR/tldr_val_results.txt"
echo $TEMP_FILE

# main evaluation function for tldr
eval_tldr() {
    mdir=$1
    model=$2
    split=$3
    set_save_dir $mdir "tldr_$split"
    mkdir -p $save_dir

    if [[ "$model" == "llama" ]]; then
        chat_formatting_function="eval.templates.create_prompt_with_tulu_chat_format"
    elif [[ "$model" == "mistral" ]]; then
        chat_formatting_function="eval.templates.create_prompt_with_llama2_chat_format"
    else
        echo "Unsupported model type"
        return 1
    fi

    if [[ "$split" == "test" ]]; then
        data_file="$DATA_DIR/tldr/test/test.jsonl"
    elif [[ "$split" == "eval" ]]; then
        data_file="$DATA_DIR/tldr/eval/tldr_val.jsonl"
    fi

    cmd="python -m eval.tldr.run_eval \
    --data_file $data_file \
    --max_context_length 1900 \
    --save_dir $save_dir \
    --model $mdir \
    --tokenizer $mdir \
    --eval_batch_size 20 \
    --use_chat_format \
    --convert_to_bf16 \
    --chat_formatting_function $chat_formatting_function"
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

# Define the function to extract evaluation results for TLDR
extract_tldr() {
    mdir=$1
    set_save_dir $mdir "tldr_eval"
    # Extract the first floating point number from the file, which represents the F1 score
    result=$(grep -oP '^\d+\.\d+' $save_dir/metrics.json | head -n 1)
    echo $result
}

# Check if tldr_val_results.txt already exists
if  [ -f "$TEMP_FILE" ]; then
    echo "tldr_val_results.txt exists, finding the best checkpoint directly."
    # Find the checkpoint with the highest BLEURT score
    best_checkpoint=$(sort -k2 -n -r $TEMP_FILE | head -n 1 | awk '{print $1}')
    echo "Best checkpoint from existing results: $best_checkpoint"
else
    echo "tldr_val_results.txt does not exist, running evaluation on checkpoints."

    # Iterate through all checkpoints in the directory with the pattern 'checkpoint-*'
    for checkpoint in "$CHECKPOINT_DIR"/checkpoint-*; do
        echo "Evaluating checkpoint: $checkpoint"
        eval_tldr $checkpoint $MODEL eval

        # Extract the evaluation performance (e.g., reward score)
        reward_score=$(extract_tldr $checkpoint)

        # Save checkpoint and its reward score
        echo "$checkpoint $reward_score" >> $TEMP_FILE
    done

    # Find the checkpoint with the highest reward score
    best_checkpoint=$(sort -k2 -n -r $TEMP_FILE | head -n 1 | awk '{print $1}')

    echo "Best checkpoint: $best_checkpoint"

fi
# Run final test on the best checkpoint
eval_tldr $best_checkpoint $MODEL test

