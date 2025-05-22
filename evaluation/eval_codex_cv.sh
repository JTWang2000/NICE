#!/bin/bash
source eval.sh
# Directory containing the checkpoints
CHECKPOINT_DIR=$1  # Directory where checkpoints are stored (e.g., file_name)
MODEL=$2           # Model name (e.g., llama)
CUDA=$3


# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=$CUDA

# Temporary file to store results
TEMP_FILE="$CHECKPOINT_DIR/val_codex_results.txt"
echo $TEMP_FILE

# main evaluation function
eval_codex() {
    mdir=$1
    model=$2
    split=$3
    set_save_dir $mdir "codex_$split"
    mkdir -p $save_dir

    if [[ "$model" == "llama" ]]; then
        chat_formatting_function="eval.templates.create_prompt_with_tulu_chat_format"
        eval_batch_size=2
    elif [[ "$model" == "mistral" ]]; then
        chat_formatting_function="eval.templates.create_prompt_with_llama2_chat_format"
        eval_batch_size=4
    else
        echo "Unsupported model type"
        return 1
    fi

    if [[ "$split" == "test" ]]; then
        data_file="$DATA_DIR/codex/test/HumanEval_test.jsonl"
    elif [[ "$split" == "eval" ]]; then
        data_file="$DATA_DIR/codex/eval/HumanEval_eval.jsonl"
    fi

    cmd="python -m eval.codex_humaneval.run_eval \
    --data_file $data_file \
    --save_dir $save_dir \
    --model_name_or_path $mdir \
    --tokenizer_name_or_path $mdir \
    --eval_batch_size $eval_batch_size \
    --unbiased_sampling_size_n 200 \
    --sampling_batch_size 20 \
    --use_chat_format \
    --temperature 0.8 \
    --chat_formatting_function $chat_formatting_function"
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

# Define the function to extract validation results and calculate the sum of pass@1, pass@10, and pass@100
extract_codex() {
    mdir=$1
    set_save_dir $mdir codex_eval

    # Extract pass@1, pass@10, and pass@100 values from metrics.json
    pass1=$(grep '"pass@1"' $save_dir/metrics.json | sed 's/.*"pass@1": \([0-9.]*\).*/\1/')
    pass10=$(grep '"pass@10"' $save_dir/metrics.json | sed 's/.*"pass@10": \([0-9.]*\).*/\1/')
    pass100=$(grep '"pass@100"' $save_dir/metrics.json | sed 's/.*"pass@100": \([0-9.]*\).*/\1/')

    # Calculate the sum of pass@1, pass@10, and pass@100
    total=$(echo "$pass1 + $pass10 + $pass100" | bc)

    echo $total
}

# Check if val_codex_results.txt already exists
if [ -f "$TEMP_FILE" ]; then
    echo "val_codex_results.txt exists, finding the best checkpoint directly."
    # Find the checkpoint with the highest total score
    best_checkpoint=$(sort -k2 -n -r $TEMP_FILE | head -n 1 | awk '{print $1}')
    echo "Best checkpoint from existing results: $best_checkpoint"
else
    echo "val_codex_results.txt does not exist, running evaluation on checkpoints."

    # Remove old results file if it exists
    rm -f $TEMP_FILE

    # Iterate through all checkpoints in the directory with the pattern 'checkpoint-*'
    for checkpoint in "$CHECKPOINT_DIR"/checkpoint-*; do
        echo "Evaluating checkpoint: $checkpoint"
        # Run validation on the checkpoint
        eval_codex $checkpoint $MODEL eval

        # Extract the validation performance based on the sum of pass@1, pass@10, and pass@100
        total_score=$(extract_codex $checkpoint)

        # Save checkpoint and its total score
        echo "$checkpoint $total_score" >> $TEMP_FILE
    done

    # Find the checkpoint with the highest total score
    best_checkpoint=$(sort -k2 -n -r $TEMP_FILE | head -n 1 | awk '{print $1}')
    echo "Best checkpoint: $best_checkpoint"
fi

# Run test on the best checkpoint
eval_codex $best_checkpoint $MODEL test
