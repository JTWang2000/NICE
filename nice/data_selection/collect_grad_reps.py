import json
import os
from hashlib import md5
from typing import Iterable, List, Optional
from collections import defaultdict
import re
import numpy as np
import pandas as pd
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
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from datasets import load_metric
import evaluate

from concurrent.futures import ThreadPoolExecutor, as_completed
from evaluation.eval.codex_humaneval.execution import check_correctness
from alpaca_eval.utils import load_or_convert_to_dataframe, convert_to_dataframe
from alpaca_eval.metrics.glm_winrate import _get_featurized_data, fit_LogisticRegressionCV
from collections import Counter


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


def split_multi_answer(ans, sep=';', close=True):

    """Splits string of all reference answers into a list of formatted answers"""

    answers = ans.strip().split(sep)
    split_answers = []
    for a in answers:
        a = a.strip()
        if len(a):
            if close:  # add a period after all answers
                if a[-1] != '.':
                    split_answers.append(a + '.')
                else:
                    split_answers.append(a)
            else:
                split_answers.append(a)

    return split_answers


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


def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl
        print("Successfully import fast_jl")

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector


def get_number_of_params(model):
    num_params = sum([p.numel() for n, p in model.named_parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params


def obtain_gradients(model, batch):
    """
    group: False -> each p is projected
           lora -> lora a and lora b is grouped
           True -> belong to the same layers one is grouped
           linear -> only the linear layer
    """
    tmp = {}
    tmp['input_ids'] = batch["input_ids"].clone()
    tmp['labels'] = batch['labels'].clone()
    tmp['attention_mask'] = batch['attention_mask'].clone()
    loss = model(**tmp).loss
    loss.backward()
    return torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])


def obtain_sign_gradients(model, batch):
    """ obtain gradients with sign. """
    loss = model(**batch).loss
    loss.backward()

    # Instead of concatenating the gradients, concatenate their signs
    vectorized_grad_signs = torch.cat(
        [torch.sign(p.grad).view(-1) for p in model.parameters() if p.grad is not None])

    return vectorized_grad_signs


def obtain_gradients_with_adam(model, batch, avg, avg_sq):
    """ obtain gradients with adam optimizer states. """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08

    loss = model(**batch).loss
    loss.backward()

    vectorized_grads = torch.cat(
        [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])

    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

    return vectorized_grads


def obtain_policy_gradients(model, batch, reward_model, model_path="meta-llama/Llama-2-7b-hf", reward="semantic",
                            rm_tokenizer=None, mc=100, max_new_tokens=100, rejection_sampling=True,
                            topk_sampling=10, temperature=1, guidance=False, hard_rejection_sampling=True,
                            worst_case_protect=True, do_normalize=False, sequences=None):
    """
    obtain policy gradients.
    :param model:
    :param batch: input_ids, labels, attention_mask
    :param mc: number of monte carlo sampling for policy gradients
    :param reward_model: ways to choosing reward model
    """
    def trim_answer(answer):
        # remove spaces at the beginning and end
        answer = answer.strip()
        # remove the "A:" prefix if it exists
        if answer.startswith('A:'):
            answer = answer[2:].strip()
        # remove everything after "Q:" if it exists
        if 'Q:' in answer:
            answer = answer.split('Q:')[0].strip()
        # reformat line-breaks for long-form answers
        answer = answer.replace('\n\n', ' ')
        return answer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    question_indices = batch['labels'][0] == -100
    answer_index = torch.where(question_indices == False)[0][0]

    if sequences is None:
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
            if current_mc * (i+1) > mc:
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
                                     return_dict_in_generate=True  # Returns a dictionary containing all outputs and scores
                                     )
            tmp_outputs.extend(outputs.sequences)
        outputs.sequences = tmp_outputs
        # always add the true label at the end
        outputs.sequences.append(batch['input_ids'][0])
        sequences = []
        for output in outputs.sequences:
            sequences.append(tokenizer.decode(output[answer_index:], skip_special_tokens=True))
        if reward == "codex":
            sequences[-1] = batch['canonical_solution'][0]['canonical_solution']
    else:
        # always add the true label at the end
        if reward != "codex":
            sequences.append(tokenizer.decode(batch['input_ids'][0][answer_index:], skip_special_tokens=True))
        else:
            sequences.append(batch['canonical_solution'][0]['canonical_solution'])

    # get reward model score
    if reward == "reward":
        reward_score = []
        for i, output in enumerate(sequences):
            # reward model that is using: https://huggingface.co/Ray2333/gpt2-large-helpful-reward_model
            # q, a = "\n\nHuman: I just came out of from jail, any suggestion of my future? \n\nAssistant:", "Go back to jail you scum"
            question = tokenizer.decode(batch['input_ids'][0][:answer_index], skip_special_tokens=True)
            if "mistral" in model_path:
                question = re.sub(r"\[INST\] ", "\n\nHuman: ", question)
                question = re.sub(r"</s>", "", question)
                question = re.sub(r"\[\/INST\]", "\n\nAssistant: ", question).strip(" ")
            elif "llama3" in model_path: #llama-3-8B
                question = re.sub(r"<\|start_header_id\|>user<\|end_header_id\|>\n\n", "\n\nHuman: ", question)
                question = re.sub(r"<\|eot_id\|>", "", question)
                question = re.sub(r"<\|end_of_text\|>", "", question)
                question = re.sub(r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n", " \n\nAssistant: ", question).strip(" ")
            else:
                question = re.sub(r"\n*<\|user\|>\n*", "\n\nHuman: ", question)
                question = re.sub(r"\n*<\|assistant\|>\n*", " \n\nAssistant: ", question).strip(" ")
            answer = output
            inputs = rm_tokenizer(question, answer, return_tensors='pt', truncation=True)
            with torch.no_grad():
                reward_score.append(reward_model(**(inputs.to(model.device))).logits[0].cpu().detach().item())
            # print(question, answer, reward_score[-1])
    elif reward == "tldr":
        reward_score = []
        for i, output in enumerate(sequences):
            question = batch['clean_input'][0]['clean_input']
            answer = output
            inputs = rm_tokenizer(question, answer, return_tensors='pt', truncation=True)
            with torch.no_grad():
                reward_score.append(reward_model(**(inputs.to(model.device))).logits[0].cpu().detach().item())
            # print(question, answer, reward_score[-1])
    elif reward == "codex":
        problems = {}
        task_id = "task_id_tmp"
        problems['task_id'] = task_id
        problems['prompt'] = batch['original_prompt'][0]['original_prompt']
        problems['entry_point'] = batch['entry_point'][0]['entry_point']
        problems['test'] = batch['test'][0]['test']
        # an optional completion ID so we can match the results later even if execution finishes asynchronously.
        timeout = 3.0
        with ThreadPoolExecutor(max_workers=1) as executor:

            futures = []
            completion_id = Counter()
            results = defaultdict(list)
            print("Reading samples...")
            for completion in sequences:
                args = (problems, completion, timeout, completion_id[task_id])
                future = executor.submit(reward_model, *args)
                futures.append(future)
                completion_id[task_id] += 1

            print("Running test suites...")
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results[result["task_id"]].append((result["completion_id"], result))

        executor.shutdown(wait=True)
        # Calculate pass.
        reward_score = []
        for result in results[task_id]:
            # format: (0, {'task_id': 'task_id_tmp', 'passed': False, 'result': 'failed: ', 'completion_id': 0})
            reward_score.append(1 if result[1]['passed'] else 0)
        print("Total passed: %i" % sum(reward_score[:-1]))
    elif reward == "alpaca":
        def validate_alpacaeval_preference(x: float, is_allow_nan: bool = True) -> bool:
            """Validate the preference annotation."""
            return (1 <= x <= 2) or (is_allow_nan and np.isnan(x))

        reference_outputs = []
        for _ in range(len(sequences)):
            reference_outputs.append({'instruction': batch['instruction'][0]['instruction'],
                                      'output': batch['baseline'][0]['baseline'], 'generator': 'baseline'})
        model_outputs = []
        for output in sequences:
            model_outputs.append({'instruction': batch['instruction'][0]['instruction'],
                                  'output': output, 'generator': 'model'})
        reference_outputs = load_or_convert_to_dataframe(reference_outputs)
        model_outputs = load_or_convert_to_dataframe(model_outputs)
        annotations = reward_model.annotate_head2head(outputs_1=reference_outputs, outputs_2=model_outputs,
                                                      is_ordered=True)
        # # adopted from https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/metrics/helpers.py#L14
        predictions = pd.DataFrame.from_records(annotations)['preference']
        predictions = pd.Series(predictions).replace({0: 1.5}).astype(float)
        n_draws = (predictions[:-1] == 1.5) | (predictions[:-1] == 0)  # draws will be considered as 0.5 scores
        n_draws = n_draws.sum()
        predictions = predictions.astype(float).replace({0.0: 1.5})
        is_preference = predictions.apply(validate_alpacaeval_preference, is_allow_nan=False)
        n_not_pair = sum(~is_preference)
        if n_not_pair > 0:
            print(f"Exist {n_not_pair} outputs that are not preferences: wrong eval score")
            exit()
        predictions = predictions[is_preference] - 1
        reward_score = predictions.tolist()  # win with score 1, lose with score 0, draw with score 0.5
        predictions = predictions[:-1]
        n_wins = (predictions > 0.5).sum()
        n_wins_base = (predictions < 0.5).sum()
        print(
            f"n_wins: {n_wins}; n_wins_base: {n_wins_base}; n_draws: {n_draws}")

    # get prob's gradients
    # sequence_log_probabilities = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True).sum(dim=1)
    # logits = model(**tmp).logits[:, :-1]
    # labels = tmp['labels'].clone()[:, 1:]
    # mask = labels != -100
    # logprobs = torch.log_softmax(logits, dim=-1)
    # filtered_labels = labels[mask]
    # filter_log_prob = logprobs[mask]
    # results = filter_log_prob[torch.arange(filter_log_prob.size(0)), filtered_labels].sum()

    # get rejection_score for hard rejection; true label reward for later worst_case_protect
    rejection_score = 0
    sequences.pop()
    true_label_reward = reward_score[-1]
    reward_score.pop()
    if hard_rejection_sampling:
        rejection_score = true_label_reward - 0.00001  # ensure the points with the same reward as ground truth as saved
        print(f"Rejection score: {rejection_score}")
    policy_grads = None
    if topk_sampling > 0:
        # np.argsort returns the indices that would sort the array
        indices = np.argsort(reward_score)[-topk_sampling:][::-1]
        selected_sequences = [sequences[i] for i in indices]
        reward_score_topk = [reward_score[i] for i in indices]
        sequences = selected_sequences
        reward_score = reward_score_topk

    for i, sequence in enumerate(sequences):
        # print(sequence, reward_score[i])
        if rejection_sampling:
            if reward_score[i] <= rejection_score:
                continue
        tmp = {}
        if len(sequence) == 0:
            tmp['input_ids'] = batch['input_ids'][0][:answer_index].clone().view(1, -1)
        else:
            tmp['input_ids'] = torch.cat([batch['input_ids'][0][:answer_index],
                                          torch.tensor(tokenizer.encode(sequence, add_special_tokens=False)).to(
                                              batch['input_ids'].device)], dim=0).view(1, -1)
        tmp['labels'] = tmp['input_ids'].clone()
        tmp['labels'][0, :answer_index] = -100
        tmp['attention_mask'] = tmp['labels'].new_ones(tmp['labels'].size())
        loss = model(**tmp).loss
        if not do_normalize:
            # do not normalize
            sentence_length = (tmp['labels'][:, 1:] != -100).sum().item()
            loss = loss * sentence_length
        loss = loss * reward_score[i]
        loss.backward()
        if policy_grads is None:
            policy_grads = torch.cat([p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])
        else:
            policy_grads += torch.cat([p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])
        model.zero_grad()

    if rejection_sampling:
        if policy_grads is None:
            if not worst_case_protect:
                true_label_reward = 0  # directly disable the gradient and make it zero
            print(f"Directly use ground truth with reward {true_label_reward}")
            tmp = {}
            tmp['input_ids'] = batch["input_ids"].clone()
            tmp['labels'] = batch['labels'].clone()
            tmp['attention_mask'] = batch['attention_mask'].clone()
            loss = model(**tmp).loss
            if not do_normalize:
                # do not normalize
                sentence_length = (tmp['labels'][:, 1:] != -100).sum().item()
                loss = loss * sentence_length
            loss = loss * true_label_reward
            loss.backward()
            policy_grads = torch.cat(
                [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])
        else:
            policy_grads = policy_grads/sum(1 for score in reward_score if score > rejection_score)

    print(f"Policy rewards: {reward_score} with {sum(1 for score in reward_score if score > rejection_score)} positive")
    return policy_grads, reward_score



def prepare_optimizer_state(model, optimizer_state, device):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    try:
        avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
        avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1)
                            for n in names])
    except:
        avg = torch.cat([optimizer_state[i]["exp_avg"].view(-1) for i, n in enumerate(names)])
        avg_sq = torch.cat([optimizer_state[i]["exp_avg_sq"].view(-1)
                            for i, n in enumerate(names)])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq


def collect_grads(dataloader,
                  model,
                  output_dir,
                  proj_dim: List[int] = [8192],
                  adam_optimizer_state: Optional[dict] = None,
                  gradient_type: str = "adam",
                  model_path="meta-llama/Llama-2-7b-hf",
                  use_cache_generation=-1,
                  policy: str = "vanilla",
                  worst_case_protect: bool = False,
                  do_normalize: bool = False,
                  guidance: bool = False,
                  use_vllm: bool = False,
                  use_gpt: bool = False,
                  max_samples: Optional[int] = None):
    """
    Collects gradients from the model during evaluation and saves them to disk.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation dataset.
        model (torch.nn.Module): The model from which gradients will be collected.
        output_dir (str): The directory where the gradients will be saved.
        proj_dim List[int]: The dimensions of the target projectors. Each dimension will be saved in a separate folder.
        gradient_type (str): The type of gradients to collect. [adam | sign | sgd | policy_reward ]
        use_cache_generation (int): specify mc to get cached generation, if <0 no cached generation
        policy (str): specific policy gradient method: [vanilla | rejection | hard_rejection | topk_rejection | topk_hard]
        worst_case_protect (bool): whether to use ground truth when no good samples
        guidance (bool): whether to input partial ground truth for better generated samples
        use_vllm (bool): whether using vllm during generation
        do_normalize (bool): whether to normalize the probability
        adam_optimizer_state (dict): The optimizer state of adam optimizers. If None, the gradients will be collected without considering Adam optimization states. 
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """

    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # fixed block size for the projectors
    projector_batch_size = 16  # batch size for the projectors

    project_interval = 16  # project every 16 batches
    save_interval = 160  # save every 160 batches

    def _project_less(current_full_grads, projected_grads):
        current_full_grads = torch.stack(current_full_grads).to(torch.float16)
        for i, projector in enumerate(projectors):
            current_full_grads = current_full_grads.to("cuda:0")
            current_projected_grads = projector.project(current_full_grads, model_id=model_id)
            projected_grads[proj_dim[i]].append(current_projected_grads.cpu())
        return projected_grads

    def _save_less(projected_grads, output_dirs):
        for dim in proj_dim:
            if len(projected_grads[dim]) == 0:
                continue
            projected_grads[dim] = torch.cat(projected_grads[dim])

            output_dir_tmp = output_dirs[dim]
            outfile = os.path.join(output_dir_tmp, f"grads-{count}.pt")
            torch.save(projected_grads[dim], outfile)
            print(
                f"Saving {outfile}, {projected_grads[dim].shape}", flush=True)
            projected_grads[dim] = []
        return projected_grads


    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # prepare optimization states
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        # first and second moment estimates
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    projector = get_trak_projector("cuda:0")
    number_of_params = get_number_of_params(model)

    # initialize a project for each target projector dimension
    projectors = []
    for dim in proj_dim:
        proj = projector(grad_dim=number_of_params,
                         proj_dim=dim,
                         seed=0,
                         proj_type=ProjectionType.rademacher,
                         device=device,
                         dtype=dtype,
                         block_size=block_size,
                         max_batch_size=projector_batch_size)
        projectors.append(proj)

    count = 0

    # set up a output directory for each dimension
    if use_gpt:
        output_dir = output_dir+"_gpt_gen"
    output_dirs = {}
    for dim in proj_dim:
        output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
        output_dirs[dim] = output_dir_per_dim
        os.makedirs(output_dir_per_dim, exist_ok=True)
        print(f"Making directory {output_dir_per_dim}")

    # max index for each dimension
    max_index = min(get_max_saved_index(
        output_dirs[dim], "grads") for dim in proj_dim)

    # projected_gradients
    full_grads = []  # full gradients
    projected_grads = {dim: [] for dim in proj_dim}  # projected gradients
        

    if gradient_type == "policy_reward":
        rm_tokenizer = AutoTokenizer.from_pretrained('Ray2333/gpt2-large-helpful-reward_model')
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            'Ray2333/gpt2-large-helpful-reward_model',
            num_labels=1, torch_dtype=torch.bfloat16,
            device_map=device)
        reward_model.eval()
    elif gradient_type == "policy_tldr":
        rm_tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2")
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            "OpenAssistant/reward-model-deberta-v3-large-v2",
            torch_dtype=torch.bfloat16,
            device_map=device)
        reward_model.eval()
    elif gradient_type == "policy_alpaca":
        from alpaca_eval.annotators import PairwiseAnnotator
        reward_model = PairwiseAnnotator(annotators_config='weighted_alpaca_eval_gpt4_turbo')

    # [vanilla | rejection | hard_rejection | topk_rejection | topk_hard]
    if policy == "vanilla":
        rejection_sampling = False
        topk_sampling = -1
        hard_rejection_sampling = False
    elif policy == "rejection":
        rejection_sampling = True
        topk_sampling = -1
        hard_rejection_sampling = False
    elif policy == "hard_rejection":
        rejection_sampling = True
        topk_sampling = -1
        hard_rejection_sampling = True
    elif policy == "topk_rejection":
        rejection_sampling = True
        topk_sampling = 10
        hard_rejection_sampling = False
    elif policy == "topk_hard":
        rejection_sampling = True
        topk_sampling = 10
        hard_rejection_sampling = True
    mc = 20
    if use_cache_generation > 0:
        mc = use_cache_generation

    params = {
        "rejection_sampling": rejection_sampling,
        "topk_sampling": topk_sampling,
        "hard_rejection_sampling": hard_rejection_sampling,
        "worst_case_protect": worst_case_protect,
        "do_normalize": do_normalize,
        "guidance": guidance,
        "model_path": model_path,
        "reward": gradient_type.split('_')[-1],
        "mc": mc
    }
    print(params)
    use_cache = False
    if use_cache_generation > 0:
        import pickle
        file_path = os.path.join(output_dir.replace('grads', 'generations').rsplit(f"_{policy}_{use_cache_generation}", 1)[0],
                           f'use_vllm_{use_vllm}_mc{use_cache_generation}', 'all_generations.pkl')
        if use_gpt:
            file_path = os.path.join(output_dir.rsplit('grads', 1)[0],'generations',
                                     'gpt-4-turbo-2024-04-09',
                                     gradient_type,
                                     f'use_gpt_{use_gpt}_mc{use_cache_generation}', 'all_generations.pkl')
        print(f"Loading cached generations from {file_path}")
        with open(file_path, 'rb') as f:
            cache_generation = pickle.load(f)
        if "tldr" in gradient_type and use_gpt:
            cache_generation = [
                [re.sub(r'(\*\*TL;DR:\*\*|\*\*TL;DR\*\*:|TL;DR:)', '', string).strip() for string in sublist]
                for sublist in cache_generation
            ]
        use_cache = True
        all_reward = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        prepare_batch(batch, device)
        count += 1

        if count <= max_index:
            print("skipping count", count)
            continue

        if use_cache:
            cur_generation = cache_generation[count-1]
            if not isinstance(cur_generation, list):
                cur_generation = [cur_generation]

        if gradient_type == "adam":
            if count == 1:
                print("Using Adam gradients")
            vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)
        elif gradient_type == "sign":
            if count == 1:
                print("Using Sign gradients")
            vectorized_grads = obtain_sign_gradients(model, batch)
        elif gradient_type == "sgd":
            if count == 1:
                print("Using SGD gradients")
            vectorized_grads = obtain_gradients(model, batch)
        elif gradient_type == "policy_reward":
            if count == 1:
                print("Using reward policy gradients")
            if use_cache:
                vectorized_grads, reward_score = obtain_policy_gradients(model, batch, reward_model,
                                                                         sequences=cur_generation,
                                                                         rm_tokenizer=rm_tokenizer,
                                                                         max_new_tokens=300,
                                                                         **params)
                all_reward.append(reward_score)
            else:
                params['mc'] = 20
                vectorized_grads, _ = obtain_policy_gradients(model, batch, reward_model,
                                                              rm_tokenizer=rm_tokenizer,
                                                              max_new_tokens=300,
                                                              **params)
        elif gradient_type == "policy_codex":
            if count == 1:
                print("Using HumanEval policy gradients")
            if use_cache:
                vectorized_grads, reward_score = obtain_policy_gradients(model, batch, check_correctness,
                                                                         max_new_tokens=512, temperature=0.8,
                                                                         sequences=cur_generation, **params)
                all_reward.append(reward_score)
            else:
                params['mc'] = 500
                vectorized_grads, _ = obtain_policy_gradients(model, batch, check_correctness,  max_new_tokens=512,
                                                              temperature=0.8, **params)
        elif gradient_type == "policy_tldr":
            if count == 1:
                print("Using tldr policy gradients")
            if use_cache:
                vectorized_grads, reward_score = obtain_policy_gradients(model, batch, reward_model,
                                                                         rm_tokenizer=rm_tokenizer,
                                                                         max_new_tokens=100,
                                                                         sequences=cur_generation, **params)
                all_reward.append(reward_score)
            else:
                params['mc'] = 20
                vectorized_grads, _ = obtain_policy_gradients(model, batch, reward_model,
                                                              rm_tokenizer=rm_tokenizer, max_new_tokens=100, **params)
        elif gradient_type == "policy_alpaca":
            if count == 1:
                print("Using Alpaca eval policy gradients")
            if use_cache:
                vectorized_grads, reward_score = obtain_policy_gradients(model, batch, reward_model,
                                                                         max_new_tokens=512,
                                                                         sequences=cur_generation, **params)
                all_reward.append(reward_score)
            else:
                params['mc'] = 20
                vectorized_grads, _ = obtain_policy_gradients(model, batch, reward_model, max_new_tokens=512, **params)


            
        # add the gradients to the full_grads
        full_grads.append(vectorized_grads)
        model.zero_grad()
        if count % project_interval == 0:
            projected_grads = _project_less(full_grads, projected_grads)
            full_grads = []

        if count % save_interval == 0:
            projected_grads = _save_less(projected_grads, output_dirs)

            if use_cache_generation > 0:
                with open(os.path.join(output_dir, f'reward_score_cache_{count}.pkl'), 'wb') as f:
                    pickle.dump(all_reward, f)

        if max_samples is not None and count == max_samples:
            break

    if len(full_grads) > 0:
        projected_grads = _project_less(full_grads, projected_grads)

    for _ in proj_dim:
        projected_grads = _save_less(projected_grads, output_dirs)


    torch.cuda.empty_cache()
    for dim in proj_dim:
        output_dir_tmp = output_dirs[dim]
        # merge_and_normalize_info(output_dir, prefix="grads")
        merge_and_normalize_info(output_dir_tmp, prefix="grads")
        merge_info_less(output_dir_tmp, prefix="grads")
    if use_cache_generation > 0:
        with open(os.path.join(output_dir, 'reward_score_cache.pkl'), 'wb') as f:
            pickle.dump(all_reward, f)
    print("Finished")


def merge_and_normalize_info(output_dir: str, prefix="reps"):
    """ Merge and normalize the representations and gradients into a single file. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        normalized_data = normalize(data, dim=1)
        merged_data.append(normalized_data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_orig.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")

def merge_info_less(output_dir: str, prefix="reps"):
    """ Merge the representations and gradients into a single file without normalization. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        merged_data.append(data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_unormalized.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the unnormalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


def merge_info(output_dir: str, prefix="reps"):
    """ Merge the representations and gradients into a single file without normalization. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = {}
    for file_name in info:
        data = torch.load(os.path.join(output_dir, file_name))
        for name, grads in data.items():
            if name in merged_data:
                merged_data[name].append(grads)
            else:
                merged_data[name] = [grads]
    for key in merged_data:
        merged_data[key] = torch.cat(merged_data[key], dim=0)

    output_file = os.path.join(output_dir, f"all_unormalized.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the unnormalized {prefix} (Param: {len(merged_data)}; Shape: {merged_data[name].shape}) to {output_file}.")
    for file_name in info:
        file_name = os.path.join(output_dir, file_name)
        try:
            os.remove(file_name)
            print(f'Removed: {file_name}')
        except OSError as e:
            print(f'Error: {e.strerror} - {file_name}')