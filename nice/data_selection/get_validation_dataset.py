import json
import os
from typing import List, Tuple

import pandas as pd
import torch
from datasets import Dataset
import datasets
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase
from nice.data_selection.get_training_dataset import get_training_dataset
from copy import deepcopy
from evaluation.eval.codex_humaneval.data import read_problems
from importlib import import_module
from evaluation.eval.tldr.presets import tldr_zero_shot_context
from alpaca_eval.utils import load_or_convert_to_dataframe, convert_to_dataframe

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"


def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function

def tokenize(tokenizer: PreTrainedTokenizerBase,
             query: str,
             completion: str,
             max_length: int,
             print_ex: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Formats a chat conversation into input tensors for a transformer model.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to encode the input.
        query (str): The question part of the chat conversation.
        completion (str): The answer part of the chat conversation.
        max_length (int): The maximum length of the input tensors.
        print_ex (bool, optional): Whether to print the example. Defaults to False.

    Returns:
        tuple: A tuple containing the full input IDs, labels, and attention mask tensors.
    """
    full_prompt = query + completion

    if print_ex:
        print("******** Example starts ********")
        print(full_prompt)
        print("******** Example ends ********")

    prompt_input_ids = torch.tensor(
        tokenizer.encode(query, max_length=max_length))
    full_input_ids = torch.tensor(
        tokenizer.encode(full_prompt, max_length=max_length))
    labels = torch.tensor(tokenizer.encode(full_prompt, max_length=max_length))
    labels[:len(prompt_input_ids)] = -100
    attention_mask = [1] * len(full_input_ids)

    return full_input_ids, labels, attention_mask

def get_rlhf_dataset_raw(file,
                         chat_format="tulu",
                         sen_bert_format: str = "question"):
    dataset = {"text": []}
    with open(file) as fin:
        examples = json.load(fin)
    if chat_format == "tulu":
        chat_formatting_function = dynamic_import_function(
            "evaluation.eval.templates.create_hhrlhf_prompt_with_tulu_chat_format")
    elif chat_format == "llama":
        chat_formatting_function = dynamic_import_function(
            "evaluation.eval.templates.create_hhrlhf_prompt_with_llama2_chat_format")
    elif chat_format == "llama3":
        chat_formatting_function = dynamic_import_function(
            "evaluation.eval.templates.create_hhrlhf_prompt_with_llama3_chat_format")
    for example in examples:
        prompt = example["messages"]
        prompt = chat_formatting_function(prompt, add_bos=False)
        answer = example["messages"][-1]["content"]
        if sen_bert_format == 'question':
            dataset['text'].append(prompt)
        elif sen_bert_format == 'questionwithanswer':
            dataset['text'].append(f"{prompt}{answer}")
        elif sen_bert_format == 'sepquestionanswer':
            dataset['text'].append([prompt, answer])
        else:
            raise ValueError("Invalid sen-bert format")
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_rlhf_dataset(data_dir: str,
                     tokenizer: PreTrainedTokenizerBase,
                     max_length: int,
                     use_chat_format=True,
                     chat_format="tulu",
                     keep_raw: bool = False,
                     sen_bert_format: str = "question",
                     **kwargs
                     ):
    """
    Get the RLHF dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <|assistant|>

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the prompts. Defaults to True.
        chat_format (str, optional): The chat format to use for the prompts. Defaults to "tulu".

    Returns:
        Dataset: The tokenized dataset containing input_ids, attention_mask, and labels.
    """
    file_name = "hh_rlhf_validation_data.jsonl"
    file = os.path.join(f"{data_dir}/eval/hh_rlhf/eval", file_name)
    if keep_raw:
        dataset = get_rlhf_dataset_raw(file, chat_format, sen_bert_format)
    else:
        if chat_format == "tulu":
            func_name = "encode_with_final_messages_format"
        elif chat_format == "llama":
            func_name = "encode_with_final_messages_format_with_llama2_chat"
        elif chat_format == "llama3":
            func_name = "encode_with_final_messages_format_with_llama3_chat"
        dataset = get_training_dataset(file, tokenizer, max_length, sample_percentage=1.0,
                                       func_name=func_name)
        columns = deepcopy(dataset.column_names)
        columns.remove("input_ids")
        columns.remove("labels")
        columns.remove("attention_mask")
        dataset = dataset.remove_columns(columns)
    return dataset


def get_codex_dataset(data_dir: str,
                      tokenizer: PreTrainedTokenizerBase,
                      max_length: int,
                      use_chat_format=True,
                      chat_format="tulu",
                      add_instruction=False,
                      keep_raw: bool = False,
                      sen_bert_format: str = "question",
                      **kwargs):
    """
    Get the codex dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    Complete the following python function.\n\n\n
    <Task Prompt: start of the function>
    <|assistant|>
    Here is the completed function:\n\n\n
    <Task Prompt: start of the function>

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the prompts. Defaults to True.
        chat_format (str, optional): The chat format to use for the prompts. Defaults to "tulu".

    Returns:
        Dataset: The tokenized dataset containing input_ids, attention_mask, and labels.
    """

    # def _read_problems(evalset_file: str):
    #     return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

    def _create_prompt_with_tulu_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
        formatted_text = ""
        for message in messages:
            if message["role"] == "system":
                formatted_text += "<|system|>\n" + message["content"] + "\n"
            elif message["role"] == "user":
                formatted_text += "<|user|>\n" + message["content"] + "\n"
            elif message["role"] == "assistant":
                formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
            else:
                raise ValueError(
                    "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                        message["role"])
                )
        formatted_text += "<|assistant|>\n"
        formatted_text = bos + formatted_text if add_bos else formatted_text
        return formatted_text

    def _create_prompt_with_llama2_chat_format(messages, eos="</s>"):
        '''
        This function is adapted from the official llama2 chat completion script and LESS implementation:
        https://github.com/facebookresearch/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/generation.py#L274
        https://github.com/princeton-nlp/LESS/blob/main/less/data_selection/get_training_dataset.py
        '''
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        B_INST, E_INST = "[INST]", "[/INST]"
        formatted_text = ""
        # If you want to include system prompt, see this discussion for the template: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/discussions/4
        # However, see here that removing the system prompt actually reduce the false refusal rates: https://github.com/facebookresearch/llama/blob/main/UPDATES.md?utm_source=twitter&utm_medium=organic_social&utm_campaign=llama2&utm_content=text#observed-issue
        if messages[0]["role"] == "system":
            assert len(messages) >= 2 and messages[1][
                "role"] == "user", "LLaMa2 chat cannot start with a single system message."
            messages = [{
                "role": "user",
                "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"]
            }] + messages[2:]
        for message in messages:
            if message["role"] == "user":
                formatted_text += f"{B_INST} {(message['content']).strip()} {E_INST}"
            elif message["role"] == "assistant":
                formatted_text += f"{(message['content'])}{eos} "
            else:
                raise ValueError(
                    "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                        message["role"])
                )

        # tokenization automatically add a bos at the start of sentence, no need to introduce here
        formatted_text = formatted_text.strip()
        return formatted_text

    codex_data_dir = os.path.join(data_dir, "eval", "codex")


    if keep_raw:
        dataset = {"text": []}
    else:
        dataset = {"input_ids": [], "attention_mask": [], "labels": [],
                   "original_prompt": [], "entry_point": [], "test": [],
                   'canonical_solution': []}

    dev_questions = list(read_problems(os.path.join(codex_data_dir, "eval",  "HumanEval_eval.jsonl")).values())
    prompts = []

    if use_chat_format:
        for example in dev_questions:
            messages = [{"role": "user", "content": "Complete the following python function.\n\n\n" + example["prompt"]}]
            if chat_format == "tulu":
                prompt = _create_prompt_with_tulu_chat_format(messages, add_bos=False)
            else:
                prompt = _create_prompt_with_llama2_chat_format(messages)
            if prompt[-1] in ["\n", " "]:
                prompt += "Here is the completed function:\n\n\n" + example["prompt"]
            else:
                prompt += " Here is the completed function:\n\n\n" + example["prompt"]
            prompts.append(prompt)
    else:
        if add_instruction:
            prompts = ["Complete the following python function to return only the function body (completion). Do not include the function header or docstring.\n\n\n" + example["prompt"]
                       for example in dev_questions]
        else:
            prompts = [example["prompt"] for example in dev_questions]

    for i, prompt in enumerate(prompts):
        answer = dev_questions[i]['canonical_solution']
        if keep_raw:
            if sen_bert_format == 'question':
                dataset['text'].append(prompt)
            elif sen_bert_format == 'questionwithanswer':
                dataset['text'].append(
                    f"Question: {prompt}\n\nAnswer: {answer}")
            elif sen_bert_format == 'sepquestionanswer':
                dataset['text'].append([question, answer])
            else:
                raise ValueError("Invalid sen-bert format")
        else:
            full_input_ids, labels, attention_mask = tokenize(
                tokenizer, prompt, answer, max_length, print_ex=True if i == 0 else False)
            dataset["input_ids"].append(full_input_ids)
            dataset["labels"].append(labels)
            dataset["attention_mask"].append(attention_mask)
            dataset["original_prompt"].append(dev_questions[i]['prompt'])
            dataset["entry_point"].append(dev_questions[i]["entry_point"])
            dataset["canonical_solution"].append(dev_questions[i]["canonical_solution"])
            dataset["test"].append(dev_questions[i]["test"])
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_tldr_dataset(data_dir: str,
                     tokenizer: PreTrainedTokenizerBase,
                     max_length: int,
                     use_chat_format=True,
                     chat_format="tulu",
                     zero_shot=False,
                     keep_raw: bool = False,
                     sen_bert_format: str = "question",
                     add_instruction: bool = False,
                     **kwargs):
    """
    Get the TruthfulQA dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Subreddit: r/{subreddit}>
    <Title: {title}>
    <{post}>
    !<TL;DR:>
    <|assistant|>

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the prompts. Defaults to True.
        chat_format (str, optional): The chat format to use for the prompts. Defaults to "tulu".

    Returns:
        Dataset: The tokenized dataset containing input_ids, attention_mask, and labels.
    """

    def _create_prompt_with_tulu_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
        formatted_text = ""
        for message in messages:
            if message["role"] == "system":
                formatted_text += "<|system|>\n" + message["content"] + "\n"
            elif message["role"] == "user":
                formatted_text += "<|user|>\n" + message["content"] + "\n"
            elif message["role"] == "assistant":
                formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
            else:
                raise ValueError(
                    "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                        message["role"])
                )
        formatted_text += "<|assistant|>\n"
        formatted_text = bos + formatted_text if add_bos else formatted_text
        return formatted_text

    def _create_prompt_with_llama2_chat_format(messages, eos="</s>"):
        '''
        This function is adapted from the official llama2 chat completion script and LESS implementation:
        https://github.com/facebookresearch/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/generation.py#L274
        https://github.com/princeton-nlp/LESS/blob/main/less/data_selection/get_training_dataset.py
        '''
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        B_INST, E_INST = "[INST]", "[/INST]"
        formatted_text = ""
        # If you want to include system prompt, see this discussion for the template: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/discussions/4
        # However, see here that removing the system prompt actually reduce the false refusal rates: https://github.com/facebookresearch/llama/blob/main/UPDATES.md?utm_source=twitter&utm_medium=organic_social&utm_campaign=llama2&utm_content=text#observed-issue
        if messages[0]["role"] == "system":
            assert len(messages) >= 2 and messages[1][
                "role"] == "user", "LLaMa2 chat cannot start with a single system message."
            messages = [{
                "role": "user",
                "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"]
            }] + messages[2:]
        for message in messages:
            if message["role"] == "user":
                formatted_text += f"{B_INST} {(message['content']).strip()} {E_INST}"
            elif message["role"] == "assistant":
                formatted_text += f"{(message['content'])}{eos} "
            else:
                raise ValueError(
                    "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                        message["role"])
                )
        # tokenization automatically add a bos at the start of sentence, no need to introduce here
        formatted_text = formatted_text.strip()
        return formatted_text

    if zero_shot and use_chat_format:
        tokenized_padding = tokenizer.encode(tldr_zero_shot_context)
    tldr_data_dir = os.path.join(data_dir, "eval", "tldr")
    if keep_raw:
        dataset = {'text': []}
    else:
        dataset = {"input_ids": [], "attention_mask": [], "labels": [], 'clean_input': []}

    dev_questions = []
    with open(os.path.join(tldr_data_dir, "eval",  "tldr_val.jsonl"), 'r') as f:
        for line in f:
            dev_questions.append(json.loads(line))
    prompts = []
    for line in dev_questions:
        if add_instruction:
            prompts.append(
                f"Subreddit: r/{line['subreddit']}\n\nTitle: {line['title']}\n\n{line['post']}\n\nA brief summary of my post is (TL;DR):")
        else:
            prompts.append(f"Subreddit: r/{line['subreddit']}\n\nTitle: {line['title']}\n\n{line['post']}\n\nTL;DR:")
    dataset['clean_input'] = prompts.copy()
    original_max_length = max_length
    max_length = max_length - 100  # buffer for summary and format
    if use_chat_format:
        for idx, prompt in enumerate(prompts):
            tokenized_context = tokenizer.encode(prompt)
            # reduce context length to max_context_length
            if len(tokenized_context) > max_length:
                prompt = tokenizer.decode(
                    tokenized_context[:max_length])
            if zero_shot:
                # Use high quality examples as padding
                # always left side padding
                # https://github.com/openai/summarize-from-feedback/blob/700967448d10004279f138666442bf1497d0e705/summarize_from_feedback/tasks.py
                tokenized_context = tokenizer.encode(prompt)
                if len(tokenized_context) < max_length:
                    pad_amt = max_length - len(tokenized_context)
                prompt = tokenizer.decode(tokenized_padding[-pad_amt:]) + prompt

            messages = [{"role": "user", "content": prompt}]
            if chat_format == "tulu":
                prompts[idx] = _create_prompt_with_tulu_chat_format(messages, add_bos=False)
            elif chat_format == "llama":
                prompts[idx] = _create_prompt_with_llama2_chat_format(messages)
            elif chat_format == "llama3":
                prompts[idx] = _create_prompt_with_llama3_chat_format(messages)

    for i, prompt in enumerate(prompts):
        answer = dev_questions[i]['summary']
        if keep_raw:
            if sen_bert_format == 'question':
                dataset['text'].append(prompt)
            elif sen_bert_format == 'questionwithanswer':
                dataset['text'].append(f"{prompt}{answer}")
            elif sen_bert_format == 'sepquestionanswer':
                dataset['text'].append([question, answer])
            else:
                raise ValueError("Invalid sen-bert format")
        else:
            answer = dev_questions[i]['summary']
            full_input_ids, labels, attention_mask = tokenize(
                tokenizer, prompt, answer, original_max_length, print_ex=True if i == 0 else False)
            dataset["input_ids"].append(full_input_ids)
            dataset["labels"].append(labels)
            dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_alpaca_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format=True,
                       chat_format="tulu",
                       keep_raw: bool = False,
                       sen_bert_format: str = "question",
                       **kwargs):
    """
    Get the Alpaca eval dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    Instruction
    <|assistant|>

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the prompts. Defaults to True.
        chat_format (str, optional): The chat format to use for the prompts. Defaults to "tulu".

    Returns:
        Dataset: The tokenized dataset containing input_ids, attention_mask, and labels.
    """

    def _create_prompt_with_tulu_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
        formatted_text = ""
        for message in messages:
            if message["role"] == "system":
                formatted_text += "<|system|>\n" + message["content"] + "\n"
            elif message["role"] == "user":
                formatted_text += "<|user|>\n" + message["content"] + "\n"
            elif message["role"] == "assistant":
                formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
            else:
                raise ValueError(
                    "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                        message["role"])
                )
        formatted_text += "<|assistant|>\n"
        formatted_text = bos + formatted_text if add_bos else formatted_text
        return formatted_text

    def _create_prompt_with_llama2_chat_format(messages, eos="</s>"):
        '''
        This function is adapted from the official llama2 chat completion script and LESS implementation:
        https://github.com/facebookresearch/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/generation.py#L274
        https://github.com/princeton-nlp/LESS/blob/main/less/data_selection/get_training_dataset.py
        '''
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        B_INST, E_INST = "[INST]", "[/INST]"
        formatted_text = ""
        # If you want to include system prompt, see this discussion for the template: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/discussions/4
        # However, see here that removing the system prompt actually reduce the false refusal rates: https://github.com/facebookresearch/llama/blob/main/UPDATES.md?utm_source=twitter&utm_medium=organic_social&utm_campaign=llama2&utm_content=text#observed-issue
        if messages[0]["role"] == "system":
            assert len(messages) >= 2 and messages[1][
                "role"] == "user", "LLaMa2 chat cannot start with a single system message."
            messages = [{
                "role": "user",
                "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"]
            }] + messages[2:]
        for message in messages:
            if message["role"] == "user":
                formatted_text += f"{B_INST} {(message['content']).strip()} {E_INST}"
            elif message["role"] == "assistant":
                formatted_text += f"{(message['content'])}{eos} "
            else:
                raise ValueError(
                    "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                        message["role"])
                )
        # tokenization automatically add a bos at the start of sentence, no need to introduce here
        formatted_text = formatted_text.strip()
        return formatted_text

    with open(os.path.join(data_dir, "eval", "alpaca_eval", "eval",  "alpaca_eval_data_diverse.jsonl"), 'r') as file:
        alpaca_eval_data = [json.loads(line) for line in file]
    if keep_raw:
        dataset = {'text': []}
    else:
        dataset = {"input_ids": [], "attention_mask": [], "labels": [], "baseline": [], 'instruction': []}

    prompts = []
    if use_chat_format:
        for idx, prompt in enumerate(alpaca_eval_data):
            messages = [{"role": "user", "content": prompt["instruction"]}]
            if chat_format == "tulu":
                prompts.append(_create_prompt_with_tulu_chat_format(messages, add_bos=False))
            else:
                prompts.append(_create_prompt_with_llama2_chat_format(messages))
    else:
        prompts = [example["instruction"] for example in alpaca_eval_data]

    for i, prompt in enumerate(prompts):
        answer = alpaca_eval_data[i]['output']
        if keep_raw:
            if sen_bert_format == 'question':
                dataset['text'].append(prompt)
            elif sen_bert_format == 'questionwithanswer':
                dataset['text'].append(f"Question: {prompt}\n\nAnswer: {answer}")
            elif sen_bert_format == 'sepquestionanswer':
                dataset['text'].append([question, answer])
            else:
                raise ValueError("Invalid sen-bert format")
        else:
            answer = alpaca_eval_data[i]['output']
            full_input_ids, labels, attention_mask = tokenize(
                tokenizer, prompt, answer, max_length, print_ex=True if i == 0 else False)
            dataset["input_ids"].append(full_input_ids)
            dataset["labels"].append(labels)
            dataset["attention_mask"].append(attention_mask)
            dataset["baseline"].append(alpaca_eval_data[i]['output'])
            dataset["instruction"].append(alpaca_eval_data[i]['instruction'])
    dataset = Dataset.from_dict(dataset)
    return dataset

def get_dataset(task, **kwargs):
    """
    Get the dataset for the given task.

    Args:
        task_name (str): The name of the task.

    Raises:
        ValueError: If the task name is not valid.

    Returns:
        Dataset: The dataset.
    """
    if task == "hh_rlhf":
        return get_rlhf_dataset(**kwargs)
    elif task == "codex":
        return get_codex_dataset(**kwargs)
    elif task == "tldr":
        return get_tldr_dataset(**kwargs)
    elif task == "alpaca":
        return get_alpaca_dataset(**kwargs)
    else:
        raise ValueError("Invalid task name")


def get_dataloader(dataset, tokenizer, batch_size=1):
    def custom_collator(features):
        excluded_keys = {'Correct Answers', 'Incorrect Answers', 'ids', 'original_prompt', 'entry_point', 'test',
                         'canonical_solution', 'clean_input', 'baseline', 'instruction','dataset'}
        filtered_features = [{k: v for k, v in feature.items() if k not in excluded_keys} for feature in features]
        # Use the existing DataCollatorForSeq2Seq to handle the remaining features
        collated_features = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")(filtered_features)
        # Add back the skipped features in the original form (e.g., as a list of strings)
        if "Correct Answers" in features[0].keys():
            collated_features['Correct Answers'] = [{k: v for k, v in feature.items() if k == 'Correct Answers'} for
                                                    feature in features]
            collated_features['Incorrect Answers'] = [{k: v for k, v in feature.items() if k == 'Incorrect Answers'}
                                                      for feature in features]
        if "ids" in features[0].keys():
            collated_features['ids'] = [{k: v for k, v in feature.items() if k == 'ids'} for feature in features]
        if "original_prompt" in features[0].keys():
            collated_features['original_prompt'] = [{k: v for k, v in feature.items() if k == 'original_prompt'} for
                                                    feature in features]
            collated_features['entry_point'] = [{k: v for k, v in feature.items() if k == 'entry_point'}
                                                      for feature in features]
            collated_features['test'] = [{k: v for k, v in feature.items() if k == 'test'}
                                                for feature in features]
            collated_features['canonical_solution'] = [{k: v for k, v in feature.items() if k == 'canonical_solution'}
                                                      for feature in features]
        if "clean_input" in features[0].keys():
            collated_features['clean_input'] = [{k: v for k, v in feature.items() if k == 'clean_input'}
                                                      for feature in features]
        if "baseline" in features[0].keys():
            collated_features['baseline'] = [{k: v for k, v in feature.items() if k == 'baseline'}
                                                      for feature in features]
            collated_features['instruction'] = [{k: v for k, v in feature.items() if k == 'instruction'}
                                             for feature in features]
        if "dataset" in features[0].keys():
            collated_features['dataset'] = [{k: v for k, v in feature.items() if k == 'dataset'}
                                                      for feature in features]
        return collated_features

    data_collator = custom_collator
    # data_collator = DataCollatorForSeq2Seq(
    #         tokenizer=tokenizer, padding="longest")
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,  # When getting gradients, we only do this single batch process
                            collate_fn=data_collator)
    print("There are {} examples in the dataset".format(len(dataset)))
    return dataloader
