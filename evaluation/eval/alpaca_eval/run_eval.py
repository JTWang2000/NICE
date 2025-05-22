import os
import re
import json
import argparse
import logging
import random
import torch
import datasets
import pandas as pd
from alpaca_eval.annotators import PairwiseAnnotator
from alpaca_eval.metrics.glm_winrate import get_length_controlled_winrate
from alpaca_eval.metrics.helpers import AbsoluteScoringRule, ZeroOneScoringRule
from alpaca_eval.utils import load_or_convert_to_dataframe, convert_to_dataframe
from eval.utils import (query_openai_chat_model, query_openai_model, generate_completions,
                        dynamic_import_function, load_hf_lm_and_tokenizer)


def main(args):
    """
    For alpaca farm
    >>> import pandas as pd
    >>> df = pd.DataFrame({"instruction": ["solve", "write backwards", "other 1", "pad"],
    ...                    "input": ["1+1", "'abc'", "", "pad_in"]})
    >>> make_prompts(df, template="first: {instruction} {input}, second: {instruction} {input}", batch_size=2)[0]
    ["first: solve 1+1, second: write backwards 'abc'", 'first: other 1 , second: pad pad_in']
    :param args:
    :return:
    """
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
    os.makedirs(args.save_dir, exist_ok=True)

    logging.info("loading data and model...")
    # alpaca_eval_data = datasets.load_dataset("tatsu-lab/alpaca_eval")["eval"]
    alpaca_eval_data = load_or_convert_to_dataframe(args.reference_path).to_dict(orient='records')
    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    for example in alpaca_eval_data:
        prompt = example["instruction"]
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
        prompts.append(prompt)

    if args.model_name_or_path is not None:
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
            convert_to_half=args.convert_to_half,
            convert_to_bf16=args.convert_to_bf16
        )
        print(next(model.parameters()).dtype)
        eos = tokenizer.encode(
            "</s>", add_special_tokens=False)[-1]
        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.eval_batch_size if args.eval_batch_size else 1,
            stop_id_sequences=[[eos]],
        )
    else:
        import openai
        openai.api_key = "7cf72d256d55479383ab6db31cda2fae"
        openai.api_base =  "https://pnlpopenai2.openai.azure.com/"
        openai.api_type = 'azure'
        openai.api_version = '2023-05-15' # this may change in the future
        openai_query_cache_path = os.path.join(args.save_dir, "openai_query_cache.jsonl")
        openai_func = query_openai_model if args.openai_engine == "text-davinci-003" else query_openai_chat_model
        results = openai_func(
            engine=args.openai_engine,
            instances=[{"id": str(i), "prompt": prompt} for i, prompt in enumerate(prompts)],
            batch_size=args.eval_batch_size if args.eval_batch_size else 10,
            output_path=openai_query_cache_path,
            max_tokens=args.max_new_tokens,
            reuse_existing_outputs=True,
        )
        outputs = [result["output"] for result in results]

    model_name = os.path.basename(os.path.normpath(args.model_name_or_path)) if args.model_name_or_path is not None else args.openai_engine
    model_results = []
    for example, output in zip(alpaca_eval_data, outputs):
        example["output"] = output
        example["generator"] = f"{model_name}-greedy-long"
        # fout.write(json.dumps(example) + "\n")
        model_results.append(example)
    with open(os.path.join(args.save_dir, f"{model_name}-greedy-long-output.json"), "w") as fout:
        json.dump(model_results, fout, indent=4)
    model_outputs = load_or_convert_to_dataframe(model_results)
    reference_outputs = load_or_convert_to_dataframe(args.reference_path)
    annotator = PairwiseAnnotator(annotators_config=args.annotators_config)
    annotations = annotator.annotate_head2head(outputs_1=reference_outputs, outputs_2=model_outputs)
    print(f"Return {len(annotations)}s comparisons. ")
    convert_to_dataframe(annotations).to_csv(os.path.join(args.save_dir, "annotations.csv"))

    win_rate = get_length_controlled_winrate(annotations)
    print(f"Win rate: {win_rate['win_rate']}")
    print(f"N wins: {win_rate['n_wins']}")
    print(f"Base wins: {win_rate['n_wins_base']}")
    print(f"N draws: {win_rate['n_draws']}")
    print(f"length_controlled_winrate: {win_rate['length_controlled_winrate']}")
    win_rate = {k: float(v) for k, v in win_rate.items()}
    with open(os.path.join(args.save_dir, "win_rate.txt"), "w") as fout:
        json.dump(win_rate, fout, indent=4)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_path",
        type=str,
        default="data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json",
        help="Path to the reference outputs. "
             "AlpacaEval 2.0 uses gpt4_turbo for the baseline "
             "but they limit the max_tokens to 300.",
    )
    parser.add_argument(
        "--annotators_config",
        type=str,
        default="weighted_alpaca_eval_gpt4_turbo",
        help="Specific annotators. "
        "AlpacaEval 2.0 uses weighted_alpaca_eval_gpt4_turbo for the annotator ."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/alpaca_farm")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="If specified, we will use the OpenAI API to generate the predictions.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--convert_to_half",
        action="store_true",
        help="Load model in half.",
    )
    parser.add_argument(
        "--convert_to_bf16",
        action="store_true",
        help="Load model in bf16.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
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
    parser.add_argument(
        "--max_new_tokens",
        default=512,
        type=int,
        help="Max number of new tokens",
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)