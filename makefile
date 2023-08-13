FINETUNE_SCRIPT=python3 finetune_lm.py
# FINETUNE_SCRIPT=python3 finetune_lm.py -c job
MODEL=flan-t5-base
DATASET=glue-mnli
DEVICES=2
PEFT=lora-8

# Full finetuning:
# USEAGE:
# 	make MODEL=flan-t5-base DATASET=glue-cola DEVICES="[0,1,2,3]" fft
fft:
	$(FINETUNE_SCRIPT) \
		model=${MODEL} peft=nopeft model.linearize=false \
		dataset=${DATASET} \
		batch_size=16 \
		optim.optimizer.lr=1e-5 optim.optimizer.weight_decay=0 \
		trainer.max_steps=2000 trainer.devices=${DEVICES}

	$(FINETUNE_SCRIPT) \
		model=${MODEL} peft=nopeft model.linearize=false \
		dataset=${DATASET} \
		batch_size=16 \
		optim.optimizer.lr=2e-5 optim.optimizer.weight_decay=0 \
		trainer.max_steps=2000 trainer.devices=${DEVICES}


# LoRA
# USAGES:
# 	make MODEL=flan-t5-base DATASET=glue-mnli DEVICES="[0,1,2,3]" lora
lora:
	${FINETUNE_SCRIPT} \
		model=${MODEL} peft=${PEFT} model.linearize=false \
		dataset=${DATASET} \
		batch_size=16 \
		optim.optimizer.lr=3e-5 optim.optimizer.weight_decay=0 \
		trainer.max_steps=2000 trainer.devices=${DEVICES}

	${FINETUNE_SCRIPT} \
		model=${MODEL} peft=${PEFT} model.linearize=false \
		dataset=${DATASET} \
		batch_size=16 \
		optim.optimizer.lr=4e-5 optim.optimizer.weight_decay=0 \
		trainer.max_steps=2000 trainer.devices=${DEVICES}

# l_lora
# USAGES:
# 	make MODEL=flan-t5-base DATASET=glue-cola DEVICES="[0,1,2,3]" l_lora
# 	make MODEL=flan-t5-base DATASET=glue-mnli DEVICES="[0,1,2,3]" l_lora
l_lora:
	${FINETUNE_SCRIPT} \
		model=${MODEL} peft=${PEFT} model.linearize=true \
		dataset=${DATASET} \
		batch_size=16 \
		optim.optimizer.lr=3e-5 optim.optimizer.weight_decay=0 \
		trainer.max_steps=2000 trainer.devices=${DEVICES}

	${FINETUNE_SCRIPT} \
		model=${MODEL} peft=${PEFT} model.linearize=true \
		dataset=${DATASET} \
		batch_size=16 \
		optim.optimizer.lr=4e-5 optim.optimizer.weight_decay=0 \
		trainer.max_steps=2000 trainer.devices=${DEVICES}

# Examples:
# 	make DATASET=glue-sst2 DEVICES="[0,1,2,3]" finetune_task
# 	make DATASET=glue-mnli DEVICES="[0,1,2,3]" finetune_task
#   make DATASET=glue-stsb DEVICES="[0,1,2,3]" finetune_task
finetune_task: fft lora l_lora
.PHONY: finetune_task
