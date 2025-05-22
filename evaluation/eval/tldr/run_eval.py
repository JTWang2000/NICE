import argparse
import json
import os
import random

import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from eval.tldr.presets import tldr_zero_shot_context

from eval.utils import (dynamic_import_function, generate_completions,
                        load_hf_lm_and_tokenizer, query_openai_chat_model)


def main(args):
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

    print("Loading data...")

    test_data = []
    with open(args.data_file) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(
        f"Loaded {len(test_data)} examples")

    load_data = []
    for line in test_data:
        load_data.append(f"Subreddit: r/{line['subreddit']}\n\nTitle: {line['title']}\n\n{line['post']}\n\nTL;DR:")

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
            use_fast_tokenizer=not args.use_slow_tokenizer,
            convert_to_bf16=args.convert_to_bf16,
            convert_to_half=args.convert_to_half,
        )
    else:
        print("No provided model name...")
        return

    # load reward model
    # Load model directly
    device = next(model.parameters()).device
    rm_tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "OpenAssistant/reward-model-deberta-v3-large-v2",
        torch_dtype=torch.bfloat16,
        device_map=device)
    reward_model.eval()

    prompts = []
    if args.zero_shot:
        tokenized_padding = tokenizer.encode(tldr_zero_shot_context)
    if args.use_chat_format:
        print(f"Using formatting function {args.chat_formatting_function}")
        chat_formatting_function = dynamic_import_function(
            args.chat_formatting_function) if args.use_chat_format else None

        for prompt in load_data:
            tokenized_context = tokenizer.encode(prompt)
            # reduce context length to max_context_length
            if len(tokenized_context) > args.max_context_length:
                prompt = tokenizer.decode(
                    tokenized_context[:args.max_context_length])
            if args.zero_shot:
                # Use high quality examples as padding
                # always left side padding
                # https://github.com/openai/summarize-from-feedback/blob/700967448d10004279f138666442bf1497d0e705/summarize_from_feedback/tasks.py
                tokenized_context = tokenizer.encode(prompt)
                if len(tokenized_context) < args.max_context_length:
                    pad_amt = args.max_context_length - len(tokenized_context)
                prompt = tokenizer.decode(tokenized_padding[-pad_amt:]) + prompt

            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            prompts.append(prompt)

    else:
        prompts = [prompt for prompt in load_data]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    if args.model_name_or_path:
        # get the last token because the tokenizer may add space tokens at the start.
        eos = tokenizer.encode(
            "</s>", add_special_tokens=False)[-1]
        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=100,
            batch_size=args.eval_batch_size,
            stop_id_sequences=[[eos]],
        )
        # remove unnecessary space
        outputs = [output.strip() for output in outputs]
    else:
        instances = [{"id": example["id"], "prompt": prompt}
                     for example, prompt in zip(test_data, prompts)]
        results = query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            output_path=os.path.join(
                args.save_dir, "tldr_openai_predictions.jsonl"),
            batch_size=args.eval_batch_size,
        )
        outputs = [result["output"].strip().split("\n")[0].strip()
                   for result in results]

    with open(os.path.join(args.save_dir, "tldr_summary.jsonl"), "w") as fout:
        for example, output in zip(test_data, outputs):
            example["prediction_text"] = output
            fout.write(json.dumps(example) + "\n")

    print("Calculating reward model ...")
    eval_scores = []
    predictions = {}
    for i, output in enumerate(outputs):
        # question, answer = "Explain nuclear fusion like I am five", "Nuclear fusion is the process by which two or more protons and neutrons combine to form a single nucleus. It is a very important process in the universe, as it is the source of energy for stars and galaxies. Nuclear fusion is also a key process in the production of energy for nuclear power plants."
        # inputs = tokenizer(question, answer, return_tensors='pt')
        # score = rank_model(**inputs).logits[0].cpu().detach()
        question = load_data[i]  # use the cleaned format of question which is similar format as Finetuning/pretraining
        answer = output
        inputs = rm_tokenizer(question, answer, return_tensors='pt', truncation=True)
        with torch.no_grad():
            eval_scores.append(reward_model(**(inputs.to(model.device))).logits[0].cpu().detach().item())
        predictions[i] = {}
        predictions[i]['question'] = question
        predictions[i]['answer'] = answer
        predictions[i]['reward'] = eval_scores[-1]

    eval_scores_avg = np.mean(eval_scores)
    print("Average reward value: ", eval_scores_avg)

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(eval_scores_avg, fout)
        json.dump("\n" + "-"*50 + "\n", fout)
        for aprediction in predictions:
            example["prediction_text"] = output
            fout.write(json.dumps(predictions[aprediction]) + "\n")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/eval/tldr/test/test.jsonl"
    )
    parser.add_argument(
        "--max_context_length",
        # max length for model is 2048, while expected summary should be within 24-48 tokens
        # 2048 - some buffer
        type=int,
        default=1900,
        help="maximum number of tokens in the prompt."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/tldr/"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
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
        help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="Add high qulaity examples as padding. Similar to zero-shot setting."
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
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
        "--convert_to_half",
        action="store_true",
        help="Load model in half.",
    )
    parser.add_argument(
        "--convert_to_bf16",
        action="store_true",
        help="Load model in bf16.",
    )
    args = parser.parse_args()
    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (
        args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
