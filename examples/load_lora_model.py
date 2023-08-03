"""
PEFT Doc: https://huggingface.co/docs/peft/

Each 🤗 PEFT method is defined by a PeftConfig class that stores all the important parameters for building a PeftModel.
Because you’re going to use LoRA, you’ll need to load and create a LoraConfig class. Within LoraConfig, specify the following parameters:

- the task_type, or sequence-to-sequence language modeling in this case
- inference_mode, whether you’re using the model for inference or not
- r, the dimension of the low-rank matrices
- lora_alpha, the scaling factor for the low-rank matrices
- lora_dropout, the dropout probability of the LoRA layers
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel

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
