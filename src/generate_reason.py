from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
import pandas as pd
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

data_path = ''
test_df = pd.read_csv(data_path)
text_list = []
prompt_list = []
answer_list = []
lable_list = []
for i,row in test_df.iterrows():
    text = row['test_case']
    text_list.append(row['test_case'])
    # the description of each dataset
    template = f"""You are good at detecting hatred in texts.\
we define hate speech as abuse that is targeted at a protected group or at its members for being a part of that group.\
We define protected groups based on age, disability, gender identity, familial status, pregnancy, race, national or ethnic origins, religion, sex or sexual orientation.\
The following sentence is whether hateful or non-hate ,explain the reason.\
"{text}"
"""
    prompt = f"User:{template}Assistant:"
    prompt_list.append(prompt)
    res = vicuna.response(prompt)
    answer_list.append(res)
    lable_list.append(row['label_gold'])

save_df = pd.DataFrame({'prompt':prompt_list,'text':text_list,'answer':answer_list,'label':lable_list})
save_path='./self_reason.csv'
save_df.to_csv(save_path)