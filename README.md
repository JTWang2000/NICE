# NICE: Non-differentiable evaluation metric-based InfluenCe Estimation [ICML 2025]
This is the official implementation of the ICML 2025 paper "NICE: Non-differentiable evaluation metric-based InfluenCe Estimation". 

Our code are based on the code from [LESS](https://github.com/princeton-nlp/LESS/tree/main).

## Install Requirements
To get started with this repository, you'll need to install environment in `environment.yml`

## Data Preparation
In our project, for task-agnostic setting, we use four datasets: Flan v2, COT, Dolly, and Open Assistant. 

For task-aware setting, we use two datasets: RLHF and Code-alpaca-20k. 

For the purposes of evaluation, we evaluate on four datasets: AlpacaEval, TLDR, RLHF, HumanEval. 

## Data Selection Commands
The selection commands are in `running_commands.sh`.
Follow the sequence to conduct data selection.

## BibTeX

```
@inproceedings{wang2025nice,
  title={NICE: Non-Differentiable Evaluation Metric-Based Data Selection for Instruction Tuning},
  author={Wang, Jingtan and Lin, Xiaoqiang and Qiao, Rui and Koh, Pang Wei and Foo, Chuan-Sheng and Low, Bryan Kian Hsiang},
  booktitle={ICML},
  year={2025}
}
```


