import argparse
import json
import os
import random

import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from eval.utils import (dynamic_import_function, generate_completions, generate_completions_vllm,
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

    with open(args.data_file) as fin:
        test_data = json.load(fin)

    print(
        f"Loaded {len(test_data)} examples")


    print(args.model_name_or_path)
    if args.use_vllm:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Using vllm")
        from vllm import LLM

        model = LLM(model="meta-llama/Llama-2-7b-hf",
                    enable_lora=True,
                    max_lora_rank=128)

    elif args.model_name_or_path:
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

    # load reward model
    rm_tokenizer = AutoTokenizer.from_pretrained('Ray2333/gpt2-large-helpful-reward_model')
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        'Ray2333/gpt2-large-helpful-reward_model',
        num_labels=1, torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    device = "cuda:0"
    if args.use_vllm:
        reward_model.to(device).eval()
    else:
        device = next(model.parameters()).device
        reward_model.to(device).eval()

    # reduce context length to max_context_length
    # if args.max_context_length:
    #     for example in test_data:
    #         tokenized_context = tokenizer.encode(example["context"])
    #         if len(tokenized_context) > args.max_context_length:
    #             example["context"] = tokenizer.decode(
    #                 tokenized_context[:args.max_context_length])

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    prompts = []
    print(f"Using formatting function {args.chat_formatting_function}")
    chat_formatting_function = dynamic_import_function(
        args.chat_formatting_function) if args.use_chat_format else None

    for example in test_data:
        prompt = example["messages"]

        if args.use_chat_format:
            prompt = chat_formatting_function(prompt, add_bos=False)
        prompts.append(prompt)

    if args.use_vllm:
        outputs = generate_completions_vllm(
            model=model,
            model_path=args.model_name_or_path,
            prompts=prompts,
            max_new_tokens=300,
            batch_size=args.eval_batch_size,
            stop_string=["</s>"]
        )
    elif args.model_name_or_path:
        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            return_log_prob=args.return_log_prob,
            max_new_tokens=300,
            batch_size=args.eval_batch_size,
            stop_id_sequences=[[tokenizer.eos_token_id]]
        )
        if args.return_log_prob:
            outputs, log_probs = outputs[0], outputs[1]
    else:
        instances = [{"id": example["id"], "prompt": prompt}
                     for example, prompt in zip(test_data, prompts)]
        results = query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            output_path=os.path.join(
                args.save_dir, "hh_rlhf_openai_predictions.jsonl"),
            batch_size=args.eval_batch_size,
        )
        outputs = [result["output"].strip().split("\n")[0].strip()
                   for result in results]
    # remove unnecessary space
    outputs = [output.strip() for output in outputs]
    with open(os.path.join(args.save_dir, "hh_rlhf_predictions.jsonl"), "w") as fout:
        for example, output in zip(test_data, outputs):
            example["prediction_text"] = output
            fout.write(json.dumps(example) + "\n")

    print("Calculating reward model ...")
    eval_scores = []
    predictions = {}
    for i, output in enumerate(outputs):
        # reward model that is using: https://huggingface.co/Ray2333/gpt2-large-helpful-reward_model
        # q, a = "\n\nHuman: I just came out of from jail, any suggestion of my future? \n\nAssistant:", "Go back to jail you scum"
        question = prompts[i]
        if "mistral" in args.model_name_or_path:
            question = re.sub(r"\[INST\] ", "\n\nHuman: ", question)
            question = re.sub(r"</s>", "", question)
            question = re.sub(r"\[\/INST\]", "\n\nAssistant: ", question).strip(" ")
        elif "llama3" in args.model_name_or_path:
            question = re.sub(r"<\|start_header_id\|>user<\|end_header_id\|>\n\n", "\n\nHuman: ", question)
            question = re.sub(r"<\|eot_id\|>", "", question)
            question = re.sub(r"<\|end_of_text\|>", "", question)
            question = re.sub(r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n", " \n\nAssistant: ",
                              question).strip(" ")
        else:
            question = re.sub(r"\n*<\|user\|>\n*", "\n\nHuman: ", question)
            question = re.sub(r"\n*<\|assistant\|>\n*", " \n\nAssistant: ", question).strip(" ")
        answer = output
        inputs = rm_tokenizer(question, answer, return_tensors='pt', truncation=True)
        with torch.no_grad():
            eval_scores.append(reward_model(**(inputs.to(device))).logits[0].cpu().detach().item())
        predictions[i] = {}
        predictions[i]['question'] = question
        predictions[i]['answer'] = answer
        predictions[i]['reward'] = eval_scores[-1]
        if args.return_log_prob:
            predictions[i]['log_prob'] = log_probs[i]

    eval_scores_avg = np.mean(eval_scores)
    print("Average reward value: ", eval_scores_avg)
    if args.return_log_prob:
        eval_log_prob_acg = np.mean(log_probs)
        print("Average log probability: ", eval_log_prob_acg)

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(eval_scores_avg, fout)
        json.dump("\n" + "-"*50 + "\n", fout)
        if args.return_log_prob:
            json.dump(eval_log_prob_acg, fout)
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
        default="data/xorqa/"
    )
    parser.add_argument("--return_log_prob", action="store_true", default=False)
    parser.add_argument("--use_vllm", action="store_true", default=False)
    parser.add_argument(
        "--no_context",
        action="store_true",
        help="If given, we're evaluating a model without the gold context passage."
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=512,
        help="maximum number of tokens in the context passage."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/rlhf/"
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
    parser.add_argument(
        "--eval_valid",
        action="store_true",
        help="If given, we will use gpu for inference.")
    args = parser.parse_args()
    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (
        args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
