FINETUNE_SCRIPT="python3 finetune_lm.py"
# FINETUNE_SCRIPT=python3 finetune_lm.py -c job
MODEL=flan-t5-base
DEVICES="[0,1,2,3]"

for DATASET in glue-rte
do
# -----------------------------------------------------------
PEFT=lora-8

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

# L_LoRA
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

# -----------------------------------------------------------
PEFT=lora-16

# LoRA
${FINETUNE_SCRIPT} \
    model=${MODEL} peft=${PEFT} model.linearize=false \
    dataset=${DATASET} \
    batch_size=16 \
    optim.optimizer.lr=1e-5 optim.optimizer.weight_decay=0 \
    trainer.max_steps=2000 trainer.devices=${DEVICES}

${FINETUNE_SCRIPT} \
    model=${MODEL} peft=${PEFT} model.linearize=false \
    dataset=${DATASET} \
    batch_size=16 \
    optim.optimizer.lr=2e-5 optim.optimizer.weight_decay=0 \
    trainer.max_steps=2000 trainer.devices=${DEVICES}

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

# L_LoRA
${FINETUNE_SCRIPT} \
    model=${MODEL} peft=${PEFT} model.linearize=true \
    dataset=${DATASET} \
    batch_size=16 \
    optim.optimizer.lr=1e-5 optim.optimizer.weight_decay=0 \
    trainer.max_steps=2000 trainer.devices=${DEVICES}

${FINETUNE_SCRIPT} \
    model=${MODEL} peft=${PEFT} model.linearize=true \
    dataset=${DATASET} \
    batch_size=16 \
    optim.optimizer.lr=2e-5 optim.optimizer.weight_decay=0 \
    trainer.max_steps=2000 trainer.devices=${DEVICES}

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

# -----------------------------------------------------------
PEFT=lora-32

# LoRA
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

# L_LoRA
${FINETUNE_SCRIPT} \
    model=${MODEL} peft=${PEFT} model.linearize=true \
    dataset=${DATASET} \
    batch_size=16 \
    optim.optimizer.lr=1e-5 optim.optimizer.weight_decay=0 \
    trainer.max_steps=2000 trainer.devices=${DEVICES}

${FINETUNE_SCRIPT} \
    model=${MODEL} peft=${PEFT} model.linearize=true \
    dataset=${DATASET} \
    batch_size=16 \
    optim.optimizer.lr=2e-5 optim.optimizer.weight_decay=0 \
    trainer.max_steps=2000 trainer.devices=${DEVICES}

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
done
