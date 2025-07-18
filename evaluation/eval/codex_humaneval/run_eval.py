import argparse
import os
import json
import random
import torch
# import vllm
import math
from eval.utils import (
    generate_completions, 
    load_hf_lm_and_tokenizer, 
    query_openai_chat_model
)
from eval.codex_humaneval.data import write_jsonl, read_problems
from eval.codex_humaneval.evaluation import evaluate_functional_correctness


def main(args):
    import re
    match = re.search(r'seed(\d+)', args.model_name_or_path)
    if match:
        seed = int(match.group(1))
        print(f"Setting seed to be {seed}")
    else:
        seed = 42
        print(f"Setting seed to be default 42")
    random.seed(seed)
    torch.manual_seed(seed)  # Set seed for all CPU and GPU devices
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    test_data = list(read_problems(args.data_file).values())
    if args.max_num_examples is not None and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)
    print("Number of examples:", len(test_data))

    if args.use_chat_format:
        prompts = []
        chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
        for example in test_data:
            messages = [{"role": "user", "content": "Complete the following python function.\n\n\n" + example["prompt"]}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "Here is the completed function:\n\n\n" + example["prompt"]
            else:
                prompt += " Here is the completed function:\n\n\n" + example["prompt"]
            prompts.append(prompt)
    else:
        prompts = [example["prompt"] for example in test_data]
        
    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            # device map is determined by the number of gpus available.
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
        # these stop sequences are those mentioned in the codex paper.
        stop_sequences = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
        # Because many tokenizers will treat the word after space differently from the original word alone,
        # to be consistent, we add a space before tokenization and remove it after tokenization.
        stop_sequences = [tokenizer.encode(" " + x, add_special_tokens=False)[1:] for x in stop_sequences]
        outputs_per_sampling_iter = []
        for sampling_iter in range(math.ceil(args.unbiased_sampling_size_n/args.sampling_batch_size)):
            current_bsz = args.sampling_batch_size
            if current_bsz * (sampling_iter+1) > args.unbiased_sampling_size_n:
                current_bsz = args.unbiased_sampling_size_n - current_bsz * sampling_iter
            print(f"Sampling iter: {sampling_iter} / {math.ceil(args.unbiased_sampling_size_n/args.sampling_batch_size)}")
            samping_outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=512,
                batch_size=args.eval_batch_size,
                stop_id_sequences=stop_sequences,
                num_return_sequences=current_bsz,
                do_sample=True,  # if only pass@1 is evaluated, we do greedy decoding.
                top_p=0.95,
                temperature=args.temperature,
            )
            outputs_per_sampling_iter.append(samping_outputs)  # n*bsz  ;  n*bsz
        # regroup the outputs to match the number of test data.
        outputs = []
        for i in range(len(prompts)):
            for j in range(math.ceil(args.unbiased_sampling_size_n/args.sampling_batch_size)):
                current_bsz = args.sampling_batch_size
                if current_bsz * (j + 1) > args.unbiased_sampling_size_n:
                    current_bsz = args.unbiased_sampling_size_n - current_bsz * j
                for p in range(current_bsz):
                    outputs.append(outputs_per_sampling_iter[j][i * current_bsz + p])
    else:
        instances = [{
            "id": examle["task_id"], 
            "prompt": "Complete the following python function. Please only output the code for the completed function.\n\n\n" + prompt,
        } for examle, prompt in zip(test_data, prompts)]
        results = query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            output_path=os.path.join(args.save_dir, "openai_query_results.jsonl"),
            batch_size=args.eval_batch_size,
            top_p=0.95,
            temperature=args.temperature,
            n=args.unbiased_sampling_size_n,
        )
        outputs = []
        for result in results:
            for choice in result["response_metadata"]["choices"]:
                outputs.append(choice["message"]["content"])

    # duplicates test data to match the number of outputs.
    duplicate_test_data = [example for example in test_data for _ in range(args.unbiased_sampling_size_n)]
    assert len(duplicate_test_data) == len(outputs)
    predictions = [{"task_id": example["task_id"], "prompt": example["prompt"], "completion": output} for example, output in zip(duplicate_test_data, outputs)]
    prediction_save_path = os.path.join(args.save_dir, "codex_eval_predictions.jsonl")
    write_jsonl(prediction_save_path, predictions)

    pass_at_k_results = evaluate_functional_correctness(
        sample_file=prediction_save_path,
        k=args.eval_pass_at_ks,
        problems={example["task_id"]: example for example in test_data},
        n_workers=64
    )

    print(pass_at_k_results)

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(pass_at_k_results, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", 
        type=str, 
        default="data/codex_eval/HumanEval.jsonl.gz",
        help="Path to the HumanEval data file."
    )
    parser.add_argument(
        "--max_num_examples", 
        type=int, 
        default=None,
        help="Maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default=None, 
        help="If specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default=None, 
        help="If specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine", 
        type=str, 
        default=None, 
        help="If specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/codex_eval", 
        help="Directory to save the results."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--eval_pass_at_ks", 
        nargs="+", 
        type=int, 
        default=[1, 10, 100],
        help="Multiple k's that we will report pass@k."
    )
    parser.add_argument(
        "--unbiased_sampling_size_n", 
        type=int, 
        default=20,
        help="Codex HumanEval requires `n` sampled generations per prompt, to estimate the unbiased pass@k. "
    )
    parser.add_argument(
        "--sampling_batch_size",
        type=int,
        default=10,
        help="Generation batch size. "
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for sampling. This is should be low for evaluating smaller pass@k, and high for larger pass@k."
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="Load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq", 
        action="store_true", 
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true", 
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    args = parser.parse_args()
    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    assert args.unbiased_sampling_size_n >= max(args.eval_pass_at_ks), "n should be larger than the largest k in eval_pass_at_ks."
    main(args)
