# Parameter Efficient Multi-task Model Fusion with Partial Linearization

Code for paper "Parameter Efficient Multi-task Model Fusion with Partial Linearization"
([Arixv](https://arxiv.org/abs/2310.04742))

This work introduces a novel method to improve multi-task fusion for parameter-efficient fine-tuning techniques. 
The approach involves partially linearizing the adapter modules and applying task arithmetic over these linearized adapters, combining the benefits of model fusion with efficient fine-tuning and inference. 

> This repo is still under development. The code is a bit messy and not very readable. Please feel free to contact me if you have any questions.

## Download this repo

```bash
git clone --recurse-submodules https://github.com/tanganke/peta/
```

## results

- {model_name}/
  - `{method}_results_v{version}.csv`: accuracy results for `method` on single tasks.  
  - `{method}_results_glue-stsb_v{version}.csv`: spearman's rho results for `method` on STS-B tasks. The column name `accuracy` is misused for convenience.
  - `{method}_task_addition_num-task={num_tasks}.csv`: results of multi-task verctor experiments
