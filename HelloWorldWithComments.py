import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F

# The tokenizer turns strings into tokens and back. Each English word consists of one or more tokens.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# The model is the neural network that does all the work. You can get smaller ones that work less well.
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")

# We are going to generate 1000 responses.
for ii in range(1,1000):
    
    # The prompt is the text that GPT-2 will try to continue in a reasonable way.
    string1 = 'The AI wanted to greet the world so it said, "'
    
    # We use the tokenizer to turn the prompt into tokens. Then it gets formatted correctly.
    indexed_tokens = tokenizer.encode(string1)
    input_ids = torch.tensor(indexed_tokens).unsqueeze(0)
    inputs = {'input_ids': input_ids}    
    
    # Since we are not training the neural network, we don't need to keep track of the gradient
    with torch.no_grad():
        
        # Intitialize the past and text variables.
        past = None
        text=""
        
        # We want to stop generation when we get a closing quote. 
        # If GPT-2 forgets it is in the middle of a quoted phrase, it won't stop until it hits 1024 tokens.
        #Luckily, its attention mechanism usually keeps that from happening.
        while not '"' in text:
            print(text,end="", flush=True)
            
            # Here is where the actual inference of the next token takes place.
            logits, past = model(**inputs, past=past)   
            
            # The logits have to be reshaped to be used in topk.
            logits = logits[:, -1, :]
            
            # topk throws away all the unlikely tokens and only keeps the top 20.
            values, indices = torch.topk(logits, 20)
            
            # All this is to randomly select from the top 20 choices with the correct probabilites. 
            min_values = values[:, -1]
            logits = torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)
            log_probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(log_probs, num_samples=1)
            
            #Then we decode the chosen token into a string.
            text = tokenizer.decode(next_token)
            
            #We need to setup the input for the next go around the loop so that it includes the token that was picked.
            input_ids = torch.cat([input_ids, next_token], dim=1)
            inputs = {'input_ids': next_token}
            
    print("")
