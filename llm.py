# %%
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import torch

# %%
model_path = "meta-llama/Llama-2-7b-chat-hf"
file = "prompts/llama2-7b-v1.csv"

# %%
class Dataset(torch.utils.data.Dataset):
  def __init__(self, texts):
    self.texts = texts

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    return self.texts.iloc[idx]

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
  model_path,
  device_map='auto',
  load_in_4bit=True,
  trust_remote_code=True,
)

pipe = pipeline(
  'text-generation',
  model=model,
  tokenizer=tokenizer,
  max_new_tokens=512,
  temperature=0.01,
  top_p=0.9,
  repetition_penalty=1.1,
)

prompts = pd.read_csv(file, index_col=0)

answers = []
for answer in tqdm(pipe(Dataset(prompts['prompt']), return_full_text=False), total=len(prompts)):
  answers.append(answer[0]['generated_text'])

prompts['answer'] = answers

prompts.to_csv(file)

# %%



