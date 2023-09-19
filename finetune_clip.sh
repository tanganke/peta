CUDA_VISIBLE_DEVICES=7 python finetune_clip.py --config-name finetune_clip_standard model=ViT-B-16 dataset=Cars     &
CUDA_VISIBLE_DEVICES=1 python finetune_clip.py --config-name finetune_clip_standard model=ViT-B-16 dataset=DTD      &
CUDA_VISIBLE_DEVICES=2 python finetune_clip.py --config-name finetune_clip_standard model=ViT-B-16 dataset=EuroSAT  &
CUDA_VISIBLE_DEVICES=3 python finetune_clip.py --config-name finetune_clip_standard model=ViT-B-16 dataset=RESISC45 &
CUDA_VISIBLE_DEVICES=4 python finetune_clip.py --config-name finetune_clip_standard model=ViT-B-16 dataset=SVHN     &
CUDA_VISIBLE_DEVICES=5 python finetune_clip.py --config-name finetune_clip_standard model=ViT-B-16 dataset=GTSRB    &
CUDA_VISIBLE_DEVICES=6 python finetune_clip.py --config-name finetune_clip_standard model=ViT-B-16 dataset=SUN397   &

CUDA_VISIBLE_DEVICES=7 python finetune_clip.py --config-name finetune_clip_lora model=ViT-B-16 dataset=Cars     &
CUDA_VISIBLE_DEVICES=1 python finetune_clip.py --config-name finetune_clip_lora model=ViT-B-16 dataset=DTD      &
CUDA_VISIBLE_DEVICES=2 python finetune_clip.py --config-name finetune_clip_lora model=ViT-B-16 dataset=EuroSAT  &
CUDA_VISIBLE_DEVICES=3 python finetune_clip.py --config-name finetune_clip_lora model=ViT-B-16 dataset=RESISC45 &
CUDA_VISIBLE_DEVICES=4 python finetune_clip.py --config-name finetune_clip_lora model=ViT-B-16 dataset=SVHN     &
CUDA_VISIBLE_DEVICES=5 python finetune_clip.py --config-name finetune_clip_lora model=ViT-B-16 dataset=GTSRB    &
CUDA_VISIBLE_DEVICES=6 python finetune_clip.py --config-name finetune_clip_lora model=ViT-B-16 dataset=SUN397   &

CUDA_VISIBLE_DEVICES=7 python finetune_clip.py --config-name finetune_clip_l_lora model=ViT-B-16 dataset=Cars       &
CUDA_VISIBLE_DEVICES=1 python finetune_clip.py --config-name finetune_clip_l_lora model=ViT-B-16 dataset=DTD        &
CUDA_VISIBLE_DEVICES=2 python finetune_clip.py --config-name finetune_clip_l_lora model=ViT-B-16 dataset=EuroSAT    &
CUDA_VISIBLE_DEVICES=3 python finetune_clip.py --config-name finetune_clip_l_lora model=ViT-B-16 dataset=RESISC45   &
CUDA_VISIBLE_DEVICES=4 python finetune_clip.py --config-name finetune_clip_l_lora model=ViT-B-16 dataset=SVHN       &
CUDA_VISIBLE_DEVICES=5 python finetune_clip.py --config-name finetune_clip_l_lora model=ViT-B-16 dataset=GTSRB      &
CUDA_VISIBLE_DEVICES=6 python finetune_clip.py --config-name finetune_clip_l_lora model=ViT-B-16 dataset=SUN397     &
