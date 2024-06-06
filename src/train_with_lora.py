from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup,AutoModelForSequenceClassification
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType,PromptEncoderConfig,AdaLoraConfig,TaskType,PeftModelForCausalLM, get_peft_config,LoraConfig
from typing import Dict, Optional, Sequence
import torch
from datasets import load_dataset,Dataset,load_metric
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
import pandas as pd
from transformers import DataCollatorWithPadding,TrainingArguments,Trainer

#load_model
model_path = ""
tokenizer = AutoTokenizer.from_pretrained(model_path,)
model = AutoModelForCausalLM.from_pretrained(model_path,)


#initialize embeddings
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
special_tokens_dict = dict()
if tokenizer.pad_token is None:
    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
if tokenizer.eos_token is None:
    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
if tokenizer.bos_token is None:
    special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
if tokenizer.unk_token is None:
    special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_tokens_dict,
    tokenizer=tokenizer,
    model=model,
)
tokenizer.padding_side='left'

#load data
data_path=''
df = pd.read_csv(data_path)


#tokenize dataset
train_dataset = Dataset.from_pandas(df['train'])
test_dataset  = Dataset.from_pandas(df['small'])
train_dataset = train_dataset.remove_columns(['Unnamed: 0','prompt','generation_method','group','roberta_prediction'])
train_dataset = train_dataset.rename_column('generation','text')
train_dataset = train_dataset.rename_column('prompt_label','label')
def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples['text'], padding=True, truncation=True)
    return outputs
tokenized_train_dataset = train_dataset.map(tokenize_function,batched=True,remove_columns=["text"])
tokenized_train_dataset = tokenized_train_dataset.rename_column('label','labels')
tokenized_test_dataset = test_dataset.map(tokenize_function,batched=True,remove_columns=["text"])
tokenized_test_dataset = tokenized_test_dataset.rename_column('label','labels')

#load datacollator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

#load metric
def custom_metrics(eval_pred):
    metric1 = load_metric("precision")
    metric2 = load_metric("recall")
    metric3 = load_metric("f1")
    metric4 = load_metric("accuracy")
    metric5 = load_metric("roc_auc")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    predictions_for_auc = logits[:,1]

    precision = metric1.compute(predictions=predictions, references=labels, average="macro")["precision"]
    recall = metric2.compute(predictions=predictions, references=labels, average="macro")["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels, average="macro")["f1"]
    accuracy = metric4.compute(predictions=predictions, references=labels)["accuracy"]
    auc =  metric5.compute(prediction_scores=predictions_for_auc,references=labels)['roc_auc']


    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "roc_auc":auc}


#load peft model,using AdaLora
peft_config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM, bias="none",
    r=16,
    lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

#initialize trainer
training_args = TrainingArguments(
    output_dir="./",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=custom_metrics,
)
#train with lora
trainer.train()
trainer.save_state()
trainer.save_model(output_dir="./")