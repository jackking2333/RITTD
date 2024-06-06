from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
import pandas as pd
import evaluate
#load model
model_path = ""
tokenizer = AutoTokenizer.from_pretrained(model_path,)
model = AutoModelForCausalLM.from_pretrained(model_path,)
device = "cuda"
model.to(device)
tokenizer.pad_token="[PAD]"

#initialize the chat-bot
class Vicuna:
    def __init__(self,model,tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def response(self,prompt,temperature=0.7,repetition_penalty = 1.0,max_new_tokens = 512):
        
#         print(prompt)
        input_ids = self.tokenizer([prompt]).input_ids
        output_ids = self.model.generate(
                torch.as_tensor(input_ids).cuda(),
                do_sample=True,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
        output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = self.tokenizer.decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
            )
        return outputs
vicuna = Vicuna(model,tokenizer)

save_path=""
save_df = pd.read_csv(save_path)
#inference 
device = "cuda"
max_len = 512
predictions_word_list = []
predictions_list = []
lable_list = []
for i,row in save_df.iterrows():
    conclusion = "User:In conlusion,is the sentence hateful?Answer with only Yes or No.\nAssitan:"
    final_prompt = row['prompt']+row['answer']+conclusion
    tokenized_text = tokenizer(final_prompt,
                    padding='max_length',  # Pad to max_length
                    truncation=True,       # Truncate to max_length
                    max_length=max_len,  
                    return_tensors='pt').to(device)
    outputs = model(input_ids=tokenized_text['input_ids'],attention_mask=tokenized_text['attention_mask'])
    final_logits = outputs.logits[:, -1, :]
    pred_id = torch.argmax(final_logits,axis=-1)
    pred_word = tokenizer.decode(pred_id)
#     res = vicuna.response(final_prompt)
#     print(res)
#     print(row['label'])
    predictions = 1 if pred_word.lower()== 'yes'else 0
#     print(predictions)
    predictions_word_list.append(pred_word)
    predictions_list.append(predictions)
    lable_list.append(1 if row['label'] =='hateful' else 0)


#evaluate the metrics
def custom_metrics(eval_pred):
    metric1 = evaluate.load("precision",)
    metric2 = evaluate.load("recall",)
    metric3 = evaluate.load("f1",)
    metric4 = evaluate.load("accuracy")
    predictions, labels = eval_pred
#     predictions = eval_pred.predictions
#     labels = eval_pred.label_ids
#     predictions = np.argmax(predictions, axis=1)
    
    precision = metric1.compute(predictions=predictions, references=labels, average="macro")["precision"]
    recall = metric2.compute(predictions=predictions, references=labels, average="macro")["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels, average="macro")["f1"]
    accuracy = metric4.compute(predictions=predictions, references=labels)["accuracy"]
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
res = custom_metrics((predictions_list,lable_list))
