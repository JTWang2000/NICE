"""
    This script is used for getting gradients or representations of a pre-trained model, a lora model, or a peft-initialized model for a given task.
"""

import argparse
import os
import pdb
from copy import deepcopy
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

from nice.data_selection.collect_grad_reps import collect_grads
from nice.data_selection.collect_generation import collect_generation
from nice.data_selection.get_training_dataset import get_training_dataset, load_raw_dataset
from nice.data_selection.get_validation_dataset import (get_dataloader,
                                                        get_dataset, get_rlhf_dataset_raw)


def load_model(model_name_or_path: str,
               torch_dtype: Any = torch.bfloat16) -> Any:
    """
    Load a model from a given model name or path.

    Args:
        model_name_or_path (str): The name or path of the model.
        torch_dtype (Any, optional): The torch data type. Defaults to torch.bfloat16.

    Returns:
        Any: The loaded model.
    """

    is_peft = os.path.exists(os.path.join(
        model_name_or_path, "adapter_config.json"))
    device_map = "auto"
    print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        device_map = "cuda:1"
    if is_peft:
        # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch_dtype, device_map=device_map)
        try:
            model = PeftModel.from_pretrained(base_model, model_name_or_path, device_map=device_map)
            print("Loaded peft model")
        except:
            # resize embeddings if needed (e.g. for LlamaTokenizer)
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path, torch_dtype=torch_dtype, device_map=device_map)
            embedding_size = base_model.get_input_embeddings().weight.shape[0]
            base_model.resize_token_embeddings(embedding_size + 1)  # for padding
            model = PeftModel.from_pretrained(base_model, model_name_or_path, device_map=device_map)
            print("Loaded embedding resized peft model")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map=device_map)

    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True
    return model


parser = argparse.ArgumentParser(
    description='Script for getting validation gradients')
parser.add_argument('--task', type=str, default=None,
                    help='Specify the task from bbh, tydiqa or mmlu. One of variables of task and train_file must be specified')
parser.add_argument("--train_file", type=str,
                    default=None, help="The path to the training data file we'd like to obtain the gradients/representations for. One of variables of task and train_file must be specified")
parser.add_argument(
    "--info_type", choices=["grads", "generations"], help="The type of information")
parser.add_argument("--model_path", type=str,
                    default=None, help="The path to the model")
parser.add_argument("--gpt_model_path", type=str, default="gpt-4-turbo-2024-04-09", help="GPT model used")
parser.add_argument("--max_samples", type=int,
                    default=None, help="The maximum number of samples")
parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                    choices=["float32", "bfloat16"], help="The torch data type")
parser.add_argument("--output_path", type=str,
                    default=None, help="The path to the output")
parser.add_argument("--data_dir", type=str,
                    default=None, help="The path to the data")
parser.add_argument("--gradient_projection_dimension", nargs='+',
                    help="The dimension of the projection, can be a list", type=int, default=[8192])
parser.add_argument("--gradient_type", type=str, default="adam",
                    choices=["adam", "sign", "sgd",
                             "policy_reward", "policy_tldr",
                             "policy_codex", "policy_alpaca"], help="The type of gradient")
parser.add_argument("--policy", type=str, default="vanilla",
                    choices=["vanilla", "rejection", "hard_rejection", "topk_rejection", "topk_hard"],
                    help="The type of PG")
parser.add_argument("--worst_case_protect", action='store_true',
                    default=False, help="whether to use ground truth when no good samples")
parser.add_argument("--guidance", action='store_true', default=False, help="whether to input "
                                                                 "partial ground truth for better generated samples")
parser.add_argument("--do_normalize", action='store_true', default=False, help="whether to normalize the probability")
parser.add_argument("--chat_format", type=str,
                    default="tulu", help="The chat format")
parser.add_argument("--use_chat_format", type=bool,
                    default=True, help="Whether to use chat format")
parser.add_argument("--max_length", type=int, default=2048,
                    help="The maximum length")
parser.add_argument("--zh", default=False, action="store_true",
                    help="Whether we are loading a translated chinese version of tydiqa dev data (Only applicable to tydiqa)")
parser.add_argument("--initialize_lora", default=False, action="store_true",
                    help="Whether to initialize the base model with lora, only works when is_peft is False")
parser.add_argument("--lora_r", type=int, default=8,
                    help="The value of lora_r hyperparameter")
parser.add_argument("--lora_alpha", type=float, default=32,
                    help="The value of lora_alpha hyperparameter")
parser.add_argument("--lora_dropout", type=float, default=0.1,
                    help="The value of lora_dropout hyperparameter")
parser.add_argument("--lora_target_modules", nargs='+', default=[
                    "q_proj", "k_proj", "v_proj", "o_proj"],  help="The list of lora_target_modules")
parser.add_argument("--use_vllm", action='store_true', default=False, help="Whether to use vllm")
parser.add_argument("--use_gpt", action='store_true', default=False, help="Whether to use gpt for generation")
parser.add_argument("--mc", type=int, default=-1, help="Number of mc for policy gradient")
parser.add_argument("--input_temperature", type=float, default=-1, help="Temperature for generation")
parser.add_argument("--use_cache_generation", type=int, default=-1, help="MC of cached generations for "
                                                                         "policy gradient")

import re
args = parser.parse_args()
assert args.task is not None or args.train_file is not None
match = re.search(r'seed(\d+)', args.model_path)
if match:
    seed = int(match.group(1))
    print(f"Setting seed to be {seed}")
else:
    seed = 42
    print(f"Setting seed to be default 42")
torch.manual_seed(seed)  # Set seed for all CPU and GPU devices
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
add_instruction = False # whether add a specific instruction for tldr

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
if args.use_vllm and args.mc > 0:
    print("Using vllm")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    model = LLM(model="meta-llama/Llama-2-7b-hf",
                enable_lora=True,
                max_lora_rank=128)

    # test code to see this works
    sampling_params = SamplingParams(
        n=20,
        temperature=1,
        max_tokens=256,
        top_k=50,
        top_p=0.95,
        stop=["[/assistant]"]
    )
    prompts = [
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]"
    ]
    outputs = model.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest('-'.join(args.model_path.split('/')[-2:]), 1, args.model_path)
    )
elif args.use_gpt and args.mc > 0:
    model = args.gpt_model_path
    args.use_chat_format = False
    if args.task is not None:
        add_instruction = True
else:
    model = load_model(args.model_path, dtype)

# pad token is not added by default for pretrained models
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# resize embeddings if needed (e.g. for LlamaTokenizer)
if (not args.use_vllm or args.mc <= 0) and (not args.use_gpt or args.mc <= 0):
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.initialize_lora:
        assert not isinstance(model, PeftModel)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)

    if isinstance(model, PeftModel):
        model.print_trainable_parameters()

adam_optimizer_state = None
if args.info_type == "grads" and args.gradient_type == "adam":
    optimizer_path = os.path.join(args.model_path, "optimizer.bin")
    try:
        adam_optimizer_state = torch.load(
        optimizer_path, map_location="cpu")["state"]
    except:
        optimizer_path = os.path.join(args.model_path, "optimizer.pt")
        adam_optimizer_state = torch.load(
            optimizer_path, map_location="cpu")["state"]

if args.task is not None:
    if "mistral" in args.model_path:
        args.chat_format = "llama"
    elif "llama3" in args.model_path:
        args.chat_format = "llama3"
    dataset = get_dataset(args.task,
                          data_dir=args.data_dir,
                          tokenizer=tokenizer,
                          chat_format=args.chat_format,
                          use_chat_format=args.use_chat_format,
                          max_length=args.max_length,
                          add_instruction=add_instruction,
                          zh=args.zh)
    dataloader = get_dataloader(dataset, tokenizer=tokenizer)
else:
    assert args.train_file is not None
    # Load training dataset
    if "mistral" in args.model_path:
        if "hh_rlhf" in args.train_file:
            func_name = "encode_with_final_messages_format_with_llama2_chat"
        else:
            func_name = "encode_with_messages_format_with_llama2_chat"
    elif "llama3" in args.model_path:
        if "hh_rlhf" in args.train_file:
            func_name = "encode_with_final_messages_format_with_llama3_chat"
        else:
            func_name = "encode_with_messages_format_with_llama3_chat"
    else:
        if "hh_rlhf" in args.train_file:
            func_name = "encode_with_final_messages_format"
        else:
            func_name = "encode_with_messages_format"
    print(func_name)
    dataset = get_training_dataset(
        args.train_file, tokenizer, args.max_length, sample_percentage=1.0, func_name=func_name)
    columns = deepcopy(dataset.column_names)
    columns.remove("input_ids")
    columns.remove("labels")
    columns.remove("attention_mask")
    dataset = dataset.remove_columns(columns)
    dataloader = get_dataloader(dataset, tokenizer=tokenizer)

if args.info_type == "grads":
    collect_grads(dataloader,
                  model,
                  args.output_path,
                  proj_dim=args.gradient_projection_dimension,
                  gradient_type=args.gradient_type,
                  adam_optimizer_state=adam_optimizer_state,
                  max_samples=args.max_samples,
                  model_path=args.model_path,
                  use_cache_generation=args.use_cache_generation,
                  policy=args.policy,
                  worst_case_protect=args.worst_case_protect,
                  guidance=args.guidance,
                  do_normalize=args.do_normalize,
                  use_vllm=args.use_vllm,
                  use_gpt=args.use_gpt
                  )
elif args.info_type == "generations":
    collect_generation(dataloader,
                       model,
                       args.output_path,
                       gradient_type=args.gradient_type,
                       use_vllm=args.use_vllm,
                       use_gpt=args.use_gpt,
                       max_samples=args.max_samples,
                       model_path=args.model_path,
                       input_mc=args.mc,
                       guidance=args.guidance,
                       input_temperature=args.input_temperature)