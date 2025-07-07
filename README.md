# NICE Data Selection for Instruction Tuning in LLMs with Non-differentiable Evaluation Metric [ICML 2025]
This is the official implementation of the ICML 2025 paper "[NICE Data Selection for Instruction Tuning in LLMs with Non-differentiable Evaluation Metric](https://openreview.net/forum?id=2wt8m5HUBs)". 

Our code are based on the code from [LESS](https://github.com/princeton-nlp/LESS/tree/main).

## Install Requirements
To get started with this repository, you'll need to install environment in `environment.yml`

## Data Preparation
In our project, for task-agnostic setting, we use four datasets: Flan v2, COT, Dolly, and Open Assistant. 

For task-aware setting, we use two datasets: RLHF and Code-alpaca-20k. 

For the purposes of evaluation, we evaluate on four datasets: AlpacaEval, TLDR, RLHF, HumanEval. 

Dataset can be downloaded from [link](https://drive.google.com/drive/folders/15RhVMIavRa_K5GjGwFtN620w6VAlpzq8?usp=sharing).

## Data Selection Commands
The selection commands are in `running_commands.sh`.
Follow the sequence to conduct data selection.

## BibTeX

```
@inproceedings{
wang2025nice,
title={{NICE} Data Selection for Instruction Tuning in {LLM}s with Non-differentiable Evaluation Metric},
author={Jingtan Wang and Xiaoqiang Lin and Rui Qiao and Pang Wei Koh and Chuan-Sheng Foo and Bryan Kian Hsiang Low},
booktitle={Forty-second International Conference on Machine Learning},
year={2025}
}
```


