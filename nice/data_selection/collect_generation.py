import json
import os
from hashlib import md5
from typing import Iterable, List, Optional
from collections import defaultdict
import re
import time
import numpy as np
import math
import torch
import torch.nn.functional as F
from functorch import grad, make_functional_with_buffers, vmap
from peft import PeftModel
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from transformers import RobertaModel
from transformers import StoppingCriteria
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from datasets import load_metric
import evaluate
import pickle
import asyncio
from openai import AsyncOpenAI
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from evaluation.eval.codex_humaneval.execution import check_correctness
from collections import Counter

# os.environ["OPENAI_API_KEY"] = API_KEY
gpt_costs_per_thousand = {
            'gpt-4o': 0.0025,
            'gpt-4o-mini': 0.00015,
            'o1-preview': 0.015,
            'o1-mini': 0.003,
            'gpt-4-turbo': 0.01,
            'gpt-4': 0.03,
            'gpt-3.5-turbo': 0.0015,
            'gpt-3.5-turbo-instruct': 0.0015,
            'gpt-3.5-turbo-0125': 0.0005,
        }

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


def prepare_batch(batch, device=torch.device("cuda:0")):
    """ Move the batch to the device. """
    for key in batch:
        if not isinstance(batch[key], list):
            batch[key] = batch[key].to(device)


def get_max_saved_index(output_dir: str, prefix="reps") -> int:
    """
    Retrieve the highest index for which the data (either representation or gradients) has been stored.

    Args:
        output_dir (str): The output directory.
        prefix (str, optional): The prefix of the files, [reps | grads]. Defaults to "reps".

    Returns:
        int: The maximum representation index, or -1 if no index is found.
    """

    files = [file for file in os.listdir(
        output_dir) if file.startswith(prefix)]
    index = [int(file.split(".")[0].split("-")[1])
             for file in files]  # e.g., output_dir/reps-100.pt
    return max(index) if len(index) > 0 else -1


def get_output(model,
               weights: Iterable[Tensor],
               buffers: Iterable[Tensor],
               input_ids=None,
               attention_mask=None,
               labels=None,
               ) -> Tensor:
    logits = model(weights, buffers, *(input_ids.unsqueeze(0),
                                       attention_mask.unsqueeze(0))).logits
    labels = labels.unsqueeze(0)
    loss_fct = F.cross_entropy
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
    return loss


def get_number_of_params(model):
    num_params = sum([p.numel() for n, p in model.named_parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params

async def dispatch_openai_requests(
        messages_list: list[list[dict[str, Any]]],
        model: str,
        temperature: float,
        max_tokens: int,
        frequency_penalty: int,
        presence_penalty: int,
        seed: int,
        n: int,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async with AsyncOpenAI() as client:
        async_responses = [client.chat.completions.create(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            n=n
        ) for x in messages_list]
        return await asyncio.gather(*async_responses)

def gpt_get_estimated_cost(config, prompt):
    """Uses the current API costs/1000 tokens to estimate the cost of generating text from the model."""
    # Get rid of [APE] token
    prompt = prompt.replace('[APE]', '')
    # Get the number of tokens in the prompt
    n_prompt_tokens = len(prompt) // 4
    # Get the number of tokens in the generated text
    total_tokens = n_prompt_tokens + config['gpt_config']['max_tokens']
    costs_per_thousand = gpt_costs_per_thousand
    price = costs_per_thousand[config['gpt_config']['model']] * total_tokens / 1000
    return price

def confirm_cost(config, texts, n):
    total_estimated_cost = 0
    for text in texts:
        total_estimated_cost += gpt_get_estimated_cost(
            config, text) * n
    print(f"Estimated cost: ${total_estimated_cost:.2f}")
    # Ask the user to confirm in the command line
    if os.getenv("LLM_SKIP_CONFIRM") is None:
        confirm = input("Continue? (y/n) ")
        if confirm != 'y':
            raise Exception("Aborted.")

def __async_generate(config, prompt, n, seed=0):
    ml = [[{"role": "user", "content": p.replace('[APE]', '').strip()}] for p in prompt]
    answer = None

    if "text" in config['gpt_config']['model']:
        raise ValueError
    else:
        model = config['gpt_config']['model']
    print(model)

    # print('querying!!', ml)
    async def main():
        # Your code where you call dispatch_openai_requests goes here
        predictions = await asyncio.wait_for(dispatch_openai_requests(
            messages_list=ml,
            model=model,
            temperature=config['gpt_config']['temperature'],
            max_tokens=config['gpt_config']['max_tokens'],
            frequency_penalty=0,
            presence_penalty=0,
            seed=seed,
            n=n
        ), timeout=60)
        return predictions

    while answer is None:
        try:
            predictions = asyncio.run(main())
        except asyncio.TimeoutError:
            print("The task exceeded the time limit 60 s.")
        except Exception as e:
            # if 'is greater than the maximum' in str(e):
            #     raise BatchSizeException()
            print(e)
            print("Retrying....")
            time.sleep(20)

        try:
            if n > 1:
                answer = []
                for x in predictions:
                    answer.append([each_x.message.content for each_x in x.choices])
                # answer = [each_x.message.content for x in predictions for each_x in x.choices]
            else:
                answer = [x.choices[0].message.content for x in predictions]
        except Exception:
            print("Please Wait!")

    return answer


def sampling_generations_gpt(gpt_model, model_path, batch, mc=100, max_new_tokens=100,
                             temperature=1, guidance=False, needs_confirmation=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    question_indices = batch['labels'][0] == -100
    cur_prompt = batch['input_ids'][0][question_indices].view(1, -1)
    if guidance:
        sentence_length = (batch['labels'][:, 1:] != -100).sum().item()
        one_fourth_length = sentence_length // 2
        prompt_extension = batch['labels'][0][~question_indices][:one_fourth_length].view(1, -1)
        cur_prompt = torch.cat((cur_prompt, prompt_extension), dim=1)
    cur_prompt = tokenizer.decode(cur_prompt[0], skip_special_tokens=True)
    if not isinstance(cur_prompt, list):
        cur_prompt = [cur_prompt]
    print(cur_prompt)
    # HumanEval needs to use max_completion_tokens for o1-mini
    # Others use max_tokens
    config = {"batch_size": 1,
              "gpt_config": {"temperature": temperature, "max_tokens": max_new_tokens, "model": gpt_model}}
    if needs_confirmation:
        confirm_cost(config, cur_prompt, mc)
    if mc > 100:
        # GPT only accept mc=128; break into batch
        result = []
        for i in range(math.ceil(mc / 100)):
            cur_mc = 100
            if (i+1)*100 > mc:
                cur_mc = (i+1)*100 - mc
            result.extend(__async_generate(config, cur_prompt, cur_mc, seed=i)[0])
        return [result]
    else:
        return __async_generate(config, cur_prompt, mc)


def sampling_generations_vllm(model, model_path, batch, reward="reward", mc=100,
                              max_new_tokens=100, temperature=1, guidance=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    question_indices = batch['labels'][0] == -100
    # Generate multiple outputs
    stop_sequences = [tokenizer.eos_token]
    if reward == "codex":
        stop_sequences = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
    batch_size = 40
    cur_prompt = batch['input_ids'][0][question_indices].view(1, -1)
    if guidance:
        sentence_length = (batch['labels'][:, 1:] != -100).sum().item()
        one_fourth_length = sentence_length // 2
        prompt_extension = batch['labels'][0][~question_indices][:one_fourth_length].view(1, -1)
        cur_prompt = torch.cat((cur_prompt, prompt_extension), dim=1)
    cur_prompt = tokenizer.decode(cur_prompt[0], skip_special_tokens=True)
    tmp_outputs = []
    for i in range(math.ceil(mc / batch_size)):
        current_mc = batch_size
        if current_mc * (i + 1) > mc:
            current_mc = mc - (current_mc * i)
        sampling_params = SamplingParams(
            n=current_mc,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_k=50,
            top_p=0.95,
            stop=stop_sequences
        )
        outputs = model.generate(
            cur_prompt,
            sampling_params,
            lora_request=LoRARequest('-'.join(model_path.split('/')[-2:]),
                                     int(model_path.split('/')[-1].split('-')[-1]), model_path)
        )

        tmp_outputs.extend([completion_output.text for completion_output in outputs[0].outputs])
    return tmp_outputs


def sampling_generations(model, model_path, batch, reward="reward", mc=100, max_new_tokens=100, temperature=1, guidance=False):
    """
    obtain policy gradients.
    :param model:
    :param batch: input_ids, labels, attention_mask
    :param mc: number of monte carlo sampling for policy gradients
    :param reward_model: ways to choosing reward model
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    question_indices = batch['labels'][0] == -100
    # Generate multiple outputs
    stop_sequences = [[tokenizer.eos_token_id]]
    if reward == "codex":
        stop_sequences = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
        stop_sequences = [tokenizer.encode(" " + x, add_special_tokens=False)[1:] for x in stop_sequences]
    batch_size = 20
    cur_prompt = batch['input_ids'][0][question_indices].view(1, -1)
    if guidance:
        sentence_length = (batch['labels'][:, 1:] != -100).sum().item()
        one_half_length = sentence_length // 2
        prompt_extension = batch['labels'][0][~question_indices][:one_half_length].view(1, -1)
        cur_prompt = torch.cat((cur_prompt, prompt_extension), dim=1)
    tmp_outputs = []
    for i in range(math.ceil(mc / batch_size)):
        current_mc = batch_size
        if current_mc * (i + 1) > mc:
            current_mc = mc - (current_mc * i)
        outputs = model.generate(input_ids=cur_prompt,
                                 attention_mask=cur_prompt.new_ones(cur_prompt.size()),
                                 pad_token_id=tokenizer.eos_token_id,
                                 temperature=temperature,
                                 max_new_tokens=max_new_tokens,
                                 num_return_sequences=current_mc,
                                 do_sample=True,
                                 top_k=50,
                                 top_p=0.95,
                                 stopping_criteria=[KeyWordsCriteria(stop_sequences)],
                                 output_scores=True,  # Enable output of scores
                                 return_dict_in_generate=True
                                 # Returns a dictionary containing all outputs and scores
                                 )
        tmp_outputs.extend(outputs.sequences)
    outputs.sequences = tmp_outputs
    output_text = []
    answer_index = torch.where(question_indices == False)[0][0]
    for output in outputs.sequences:
        output_text.append(tokenizer.decode(output[answer_index:], skip_special_tokens=True))
    return output_text


def collect_generation(dataloader,
                       model,
                       output_dir,
                       gradient_type,
                       use_vllm=False,
                       use_gpt=False,
                       guidance=False,
                       max_samples: Optional[int] = None,
                       model_path=None,
                       input_mc=100,
                       input_temperature=-1):
    """
    Collects generated text and cache them

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation dataset.
        model (torch.nn.Module): The model from which gradients will be collected.
        output_dir (str): The directory where the gradients will be saved.
        gradient_type: decide what generation rules
        use_vllm: whether to use vllm for generation
        use_gpt: whether to use GPT for generation
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
        model_path: if use vllm, needs to provide the lora model path
    """

    save_interval = 100  # save every 100 batches

    def _save(generation_list, output_dir):
        outfile = os.path.join(output_dir, f"generations-{count}.pkl")
        with open(outfile, "wb") as f:
            pickle.dump(generation_list, f)
        print(
            f"Saving {outfile}, {len(generation_list)} with each {len(generation_list[0])}", flush=True)

    if use_vllm or use_gpt:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = next(model.parameters()).device

    # initialize a project for each target projector dimension

    count = 0
    if use_gpt:
        output_dir = os.path.join(output_dir, f"use_gpt_{use_gpt}_mc{input_mc}")
    else:
        output_dir = os.path.join(output_dir, f"use_vllm_{use_vllm}_mc{input_mc}")
    os.makedirs(output_dir, exist_ok=True)

    # max index for each dimension
    max_index = get_max_saved_index(output_dir, "generations")

    # generated text
    generation_list = []  # full gradients
    max_new_tokens = 100
    temperature = 1
    for batch in tqdm(dataloader, total=len(dataloader)):
        prepare_batch(batch, device)
        count += 1

        if count <= max_index:
            print("skipping count", count)
            continue

        if gradient_type == "policy_reward":
            if count == 1:
                print("Generating reward policy ...")
            mc = 100
            max_new_tokens = 300
            temperature = 1
        elif gradient_type == "policy_codex":
            if count == 1:
                print("Using codex policy gradients")
            mc = 500
            max_new_tokens = 512
            temperature = 0.8
        elif gradient_type == "policy_tldr":
            if count == 1:
                print("Using tldr policy gradients")
            mc = 20
            max_new_tokens = 100
            temperature = 1
        elif gradient_type == "policy_alpaca":
            if count == 1:
                print("Using alpaca eval policy gradients")
            mc = 20
            max_new_tokens = 512
            temperature = 1

        if input_mc > 0:
            mc = input_mc
        if input_temperature > 0:
            temperature = input_temperature
        if count == 1:
            print(f"mc: {mc}, temp: {temperature}; max_new_tokens: {max_new_tokens}, guidance: {guidance}")
        # else use the default mc
        if use_vllm:
            generation_list.append(sampling_generations_vllm(model, model_path, batch, mc=mc,
                                                             reward=gradient_type.split('_')[-1],
                                                             max_new_tokens=max_new_tokens,
                                                             temperature=temperature, guidance=guidance))
        elif use_gpt:
            generation_list.extend(sampling_generations_gpt(model, model_path, batch, mc=mc, max_new_tokens=max_new_tokens,
                                                            temperature=temperature, guidance=guidance))
            with open(os.path.join(output_dir, 'cache.txt'), 'a+', buffering=1) as fout:
                response_dict = {'response': generation_list[-1]}
                fout.write(json.dumps(response_dict) + '\n')
        else:
            generation_list.append(sampling_generations(model, model_path, batch, mc=mc, reward=gradient_type.split('_')[-1],
                                                        max_new_tokens=max_new_tokens,
                                                        temperature=temperature, guidance=guidance))

        if count % save_interval == 0:
            _save(generation_list, output_dir)
            generation_list = []
        if max_samples is not None and count == max_samples:
            break
    if len(generation_list) > 0:
        _save(generation_list, output_dir)
    import pdb; pdb.set_trace()

    torch.cuda.empty_cache()
    merge_info(output_dir, prefix="generations")

    print("Finished")



def merge_info(output_dir: str, prefix="reps"):
    """ Merge the representations and gradients into a single file without normalization. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        with open(os.path.join(output_dir, file), "rb") as f:
            merged_data.extend(pickle.load(f))

    output_file = os.path.join(output_dir, f"all_generations.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(merged_data, f)
    print(
        f"Saving the all {prefix} (Shape: {len(merged_data)}, {len(merged_data[0])} ) to {output_file}.")
