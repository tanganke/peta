# simple_average
FIRST_DEVICE=1
CUDA_VISIBLE_DEVICES=$(($FIRST_DEVICE))     python3 evaluate_multi_task_clip.py --method simple_average --finetune_mode standard    &
CUDA_VISIBLE_DEVICES=$(($FIRST_DEVICE+1))   python3 evaluate_multi_task_clip.py --method simple_average --finetune_mode lora        &
CUDA_VISIBLE_DEVICES=$(($FIRST_DEVICE+2))   python3 evaluate_multi_task_clip.py --method simple_average --finetune_mode l_lora      &

# task arithmetic
FIRST_DEVICE=4
CUDA_VISIBLE_DEVICES=$(($FIRST_DEVICE))     python3 evaluate_multi_task_clip.py --method task_arithmetic --finetune_mode standard  &
CUDA_VISIBLE_DEVICES=$(($FIRST_DEVICE+1))   python3 evaluate_multi_task_clip.py --method task_arithmetic --finetune_mode lora      &
CUDA_VISIBLE_DEVICES=$(($FIRST_DEVICE+2))   python3 evaluate_multi_task_clip.py --method task_arithmetic --finetune_mode l_lora    &

# ties merging
FIRST_DEVICE=1
CUDA_VISIBLE_DEVICES=$(($FIRST_DEVICE))     python3 evaluate_multi_task_clip.py --method ties_merging --finetune_mode standard  &
CUDA_VISIBLE_DEVICES=$(($FIRST_DEVICE+1))   python3 evaluate_multi_task_clip.py --method ties_merging --finetune_mode lora      &
CUDA_VISIBLE_DEVICES=$(($FIRST_DEVICE+2))   python3 evaluate_multi_task_clip.py --method ties_merging --finetune_mode l_lora    &

# ties merging
FIRST_DEVICE=4
CUDA_VISIBLE_DEVICES=$(($FIRST_DEVICE))     python3 evaluate_multi_task_clip.py --method lorahub --finetune_mode standard  &
CUDA_VISIBLE_DEVICES=$(($FIRST_DEVICE+1))   python3 evaluate_multi_task_clip.py --method lorahub --finetune_mode lora      &
CUDA_VISIBLE_DEVICES=$(($FIRST_DEVICE+2))   python3 evaluate_multi_task_clip.py --method lorahub --finetune_mode l_lora    &
