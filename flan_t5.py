from datasets import load_dataset, load_metric
from peft import LoraConfig, PeftModel, TaskType, get_peft_config, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# base model
model_name_or_path = "google/flan-t5-base"
tokenizer_name_or_path = "google/flan-t5-base"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# trainable params: 884,736 || all params: 248,462,592 || trainable%: 0.3560841867092814
tokenizer()
raw_datasets = load_dataset("glue", "mrpc")
pass


training_args = TrainingArguments(output_dir="test-trainer")
trainer = Trainer(
    model=model,
    training_args=training_args,
)
