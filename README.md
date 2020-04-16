# Hello-World-with-GPT-2
This is a very short program to get you started using the GPT-2 text generating deep network. It sometimes says "Hello, World!" but it often says other things instead.
It uses the HuggingFace Transformers library. The first time you run it, it will download the gpt2-xl neural network. This is an 11GB file, so it will take a while, even with broadband. After that it will run much faster, though it will still take a few seconds to get everything loaded into memory.

This is the full program. The key line is the prompt:

    # "Hello, World" using GPT-2
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    import torch.nn.functional as F

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    prompt = 'The AI wanted to greet the world so it said, "'
    indexed_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(indexed_tokens).unsqueeze(0)
    inputs = {'input_ids': input_ids}    
    with torch.no_grad():
        past = None
        text=""
        while not '"' in text:
            print(text,end="", flush=True)
            logits, past = model(**inputs, past=past)    
            values, indices = torch.topk(logits, 20)
            logits = logits[:, -1, :]
            log_probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(log_probs, num_samples=1)
            text = tokenizer.decode(next_token)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            inputs = {'input_ids': next_token}

Here is a sample of the output:

Hello

ace... I'm here

HELLO WORLD

Hello I am so and so

Hello

Hi! I'm chosen hero! I'm ready to travel the world and create a new world behind me

Hello, Human

Welcome. What a pretty planet

OK, let me drive around and see where I can be and get lost

I'm around

Hello
