## Evaluation

We mainly employ four evaluation datasets to assess the performance of our data selection pipeline:
**AlpacaEval**, **TLDR**, **RLHF**, **HumanEval**.
We use the evaluation pipeline [open-instruct](https://github.com/allenai/open-instruct/tree/main/eval). 
We keep a version we use to evaluate the models in `eval` folder. 
To evaluate a trained model, please check out the `eval_alpaca_cv.sh`, `eval_tldr_cv.sh`, `eval_hh_rlhf.cv`, and `eval_codex_cv.sh` scripts in the `evaluation` directory. These scripts contain the necessary commands to evaluate the model on the respective datasets. 

