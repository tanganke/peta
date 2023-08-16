# Parameter-Efficient Task Arithmetic

## results

- {model_name}/
  - `{method}_results_v{version}.csv`: accuracy results for `method` on single tasks.  
  - `{method}_results_glue-stsb_v{version}.csv`: spearman's rho results for `method` on STS-B tasks. The column name `accuracy` is misused for convenience.
  - `{method}_task_addition_num-task={num_tasks}.csv`: results of multi-task verctor experiments
