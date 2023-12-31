# -*- coding: utf-8 -*-
"""llama for chatgpt prompts.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14hRinPljdfJ7uUFl1zMaeyu5L3J5XTQ2
"""

!pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
!pip install -Uq datasets bitsandbytes einops wandb

from huggingface_hub import notebook_login

notebook_login()

import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)
import warnings

warnings.filterwarnings("ignore")

from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="fka/awesome-chatgpt-prompts", filename="prompts.csv",repo_type="dataset")

import pandas as pd
df = pd.read_csv("/root/.cache/huggingface/hub/datasets--fka--awesome-chatgpt-prompts/snapshots/7baf3f8a5f3d38acc585d42d12193b27baf8cf79/prompts.csv")
df.head()

df['text'] = f"""Below is an instruction that describes a task, Write a response that appropriately completes the request.  # noqa: E501
### Instruction:
I want to make chatgpt act as a {df["act"]}
### Response:
{df["prompt"]}"""

df.head()

!pip install datasets

from datasets import Dataset
dataset = Dataset.from_pandas(df)

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)

model_name = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

from peft import LoraConfig, get_peft_model

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

from transformers import TrainingArguments

output_dir = "./results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 50
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 300
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)

from trl import SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="prompt",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained("outputs")

lora_config = LoraConfig.from_pretrained('outputs')
model = get_peft_model(model, lora_config)

model.push_to_hub("Veer15/llama-gpt-prompter",create_pr=1)