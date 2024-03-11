MODEL=t5-base
PEFT=lora-8
DEVICES="[4,5,6,7]"

for DATASET in glue-cola glue-mnli glue-mrpc glue-qqp glue-rte glue-sst2 glue-stsb
do
    make MODEL=${MODEL} PEFT=${PEFT} DEVICES=${DEVICES} DATASET=${DATASET} finetune_task
done
