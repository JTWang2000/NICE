import torch
import tqdm
import json
import time
import asyncio
import os
from importlib import import_module
from transformers import StoppingCriteria
import math

from open_instruct.finetune import encode_with_prompt_completion_format
from eval.dispatch_openai_requests import dispatch_openai_chat_requesets, dispatch_openai_prompt_requesets


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



@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None,
                         add_special_tokens=True, disable_tqdm=False, **generation_kwargs):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")
    output_scores = False
    return_dict_in_generate = False

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt",
                                      add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        try:

            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **generation_kwargs
            )

            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break


            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        # for prompt, generation in zip(batch_prompts, batch_generations):
        #     print("========")
        #     print(prompt)
        #     print("--------")
        #     print(generation)

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"

    return generations


@torch.no_grad()
def generate_completions_vllm(model, model_path, prompts, batch_size=1, stop_string=None, disable_tqdm=False, **generation_kwargs):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")
    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    sampling_params = SamplingParams(
        n=num_return_sequences,
        temperature=1,
        max_tokens=generation_kwargs.get("max_tokens", 100),
        top_k=50,
        stop=stop_string
    )

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]

        try:
            batch_outputs = model.generate(
                batch_prompts,
                sampling_params,
                lora_request=LoRARequest('-'.join(model_path.split('/')[-2:]),1, model_path)
            )
            generations.extend(
                [completion_output.text for aoutput in batch_outputs for completion_output in aoutput.outputs])

        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            generations += [""] * len(batch_prompts) * num_return_sequences

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

        # for prompt, generation in zip(prompts, generations):
        #     print("========")
        #     print(prompt)
        #     print("--------")
        #     print(generation)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


# edit the function to support lora loading
def load_hf_lm_and_tokenizer(
        model_name_or_path, 
        tokenizer_name_or_path=None, 
        device_map="auto", 
        torch_dtype="auto",
        load_in_8bit=False, 
        convert_to_half=False,
        convert_to_bf16=True,
        gptq_model=False,
        use_fast_tokenizer=True,
        padding_side="left",
    ):
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM

    # is_peft = "lora" in model_name_or_path
    is_peft = True
    from peft import PeftConfig, PeftModel
    if is_peft:
        peft_config = PeftConfig.from_pretrained(model_name_or_path)
        peft_dir = model_name_or_path
        model_name_or_path = peft_config.base_model_name_or_path
        
    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True
        )
        model = model_wrapper.model  
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map=device_map, 
            load_in_8bit=True
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map, torch_dtype=None)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
            if torch.cuda.is_available():
                model = model.cuda()
        
        if is_peft:
            try:
                model = PeftModel.from_pretrained(model, peft_dir, device_map="auto").merge_and_unload()
                print("Loading peft model")
            except:
                # resize embeddings if needed (e.g. for LlamaTokenizer)
                if device_map:
                    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map,
                                                                 torch_dtype=None)
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
                    if torch.cuda.is_available():
                        model = model.cuda()
                embedding_size = model.get_input_embeddings().weight.shape[0]
                model.resize_token_embeddings(embedding_size + 1)  # for padding
                model = PeftModel.from_pretrained(model, peft_dir, device_map="auto").merge_and_unload()
                print("Loading embedding resized peft model")
         
        if convert_to_half:
            model = model.half()
            print("Convert model to half precision.")
            assert next(model.parameters()).dtype == torch.float16, "model parameters should be in half precision"
        elif convert_to_bf16:
            model = model.bfloat16()
            print("Convert model to bfloat16 precision.")
            assert next(model.parameters()).dtype == torch.bfloat16, "model parameters should be in bfloat16 precision"
        else:
            print("Model is not converted to half or bfloat16 precision.")
            assert next(model.parameters()).dtype == torch.float32, "model parameters should be in float32 precision"

    model.eval()

    if not tokenizer_name_or_path:
        if is_peft:
            tokenizer_name_or_path = peft_dir
        else:
            tokenizer_name_or_path = model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    except:
        # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # replace with new embeddings 
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    # for OPT and Pythia models, we need to set tokenizer.model_max_length to model.config.max_position_embeddings 
    # to avoid wrong embedding index.    
    if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
        tokenizer.model_max_length = model.config.max_position_embeddings
        print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))
        
    return model, tokenizer



def query_openai_chat_model(engine, instances, output_path=None, batch_size=10, retry_limit=5, reuse_existing_outputs=True, **completion_kwargs):
    '''
    Query OpenAI chat model and save the results to output_path.
    `instances` is a list of dictionaries, each dictionary contains a key "prompt" and a key "id".
    '''
    existing_data = {}
    if reuse_existing_outputs and output_path is not None and os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                existing_data[instance["id"]] = instance

    # by default, we use temperature 0.0 to get the most likely completion.
    if "temperature" not in completion_kwargs:
        completion_kwargs["temperature"] = 0.0

    results = []
    if output_path is not None:
        fout = open(output_path, "w")

    retry_count = 0
    progress_bar = tqdm.tqdm(total=len(instances))
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i+batch_size]
        if all([x["id"] in existing_data for x in batch]):
            results.extend([existing_data[x["id"]] for x in batch])
            if output_path is not None:
                for instance in batch:
                    fout.write(json.dumps(existing_data[instance["id"]]) + "\n")
                    fout.flush()
            progress_bar.update(batch_size)
            continue
        messages_list = []
        for instance in batch:
            messages = [{"role": "user", "content": instance["prompt"]}]
            messages_list.append(messages)
        while retry_count < retry_limit:
            try:
                outputs = asyncio.run(
                    dispatch_openai_chat_requesets(
                    messages_list=messages_list,
                    model=engine,
                    **completion_kwargs,
                ))
                retry_count = 0
                break
            except Exception as e:
                retry_count += 1
                print(f"Error while requesting OpenAI API.")
                print(e)
                print(f"Sleep for {30*retry_count} seconds.")
                time.sleep(30*retry_count)
                print(f"Retry for the {retry_count} time.")
        if retry_count == retry_limit:
            raise RuntimeError(f"Failed to get response from OpenAI API after {retry_limit} retries.")
        assert len(outputs) == len(batch)
        for instance, output in zip(batch, outputs):
            instance[f"output"] = output["choices"][0]["message"]["content"]
            instance["response_metadata"] = output
            results.append(instance)
            if output_path is not None:
                fout.write(json.dumps(instance) + "\n")
                fout.flush()
        progress_bar.update(batch_size)
    return results
 

def query_openai_model(engine, instances, output_path=None, batch_size=10, retry_limit=5, reuse_existing_outputs=True, **completion_kwargs):
    '''
    Query OpenAI chat model and save the results to output_path.
    `instances` is a list of dictionaries, each dictionary contains a key "prompt" and a key "id".
    '''
    existing_data = {}
    if reuse_existing_outputs and output_path is not None and os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                existing_data[instance["id"]] = instance

    # by default, we use temperature 0.0 to get the most likely completion.
    if "temperature" not in completion_kwargs:
        completion_kwargs["temperature"] = 0.0

    results = []
    if output_path is not None:
        fout = open(output_path, "w")

    retry_count = 0
    progress_bar = tqdm.tqdm(total=len(instances))
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i+batch_size]
        if all([x["id"] in existing_data for x in batch]):
            results.extend([existing_data[x["id"]] for x in batch])
            if output_path is not None:
                for instance in batch:
                    fout.write(json.dumps(existing_data[instance["id"]]) + "\n")
                    fout.flush()
            progress_bar.update(batch_size)
            continue
        messages_list = []
        for instance in batch:
            messages = instance["prompt"]
            messages_list.append(messages)
        while retry_count < retry_limit:
            try:
                outputs = asyncio.run(
                    dispatch_openai_prompt_requesets(
                    prompt_list=messages_list,
                    model=engine,
                    **completion_kwargs,
                ))
                retry_count = 0
                break
            except Exception as e:
                retry_count += 1
                print(f"Error while requesting OpenAI API.")
                print(e)
                print(f"Sleep for {30*retry_count} seconds.")
                time.sleep(30*retry_count)
                print(f"Retry for the {retry_count} time.")
        if retry_count == retry_limit:
            raise RuntimeError(f"Failed to get response from OpenAI API after {retry_limit} retries.")
        assert len(outputs) == len(batch)
        for instance, output in zip(batch, outputs):
            instance[f"output"] = output["choices"][0]["text"]
            instance["response_metadata"] = output
            results.append(instance)
            if output_path is not None:
                fout.write(json.dumps(instance) + "\n")
                fout.flush()
        progress_bar.update(batch_size)
    return results


def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function
 