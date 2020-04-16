import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
for ii in range(1,1000):
    string1 = 'The AI wanted to greet the world so it said, "'
    indexed_tokens = tokenizer.encode(string1)
    input_ids = torch.tensor(indexed_tokens).unsqueeze(0)
    inputs = {'input_ids': input_ids}    
    with torch.no_grad():
        past = None
        text=""
        while not '"' in text:
            print(text,end="", flush=True)
            logits, past = model(**inputs, past=past)   
            logits = logits[:, -1, :]
            values, indices = torch.topk(logits, 20)
            min_values = values[:, -1]
            logits = torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)
            log_probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(log_probs, num_samples=1)
            text = tokenizer.decode(next_token)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            inputs = {'input_ids': next_token}
    print("")
                
           
    
    
    
    
    
    
