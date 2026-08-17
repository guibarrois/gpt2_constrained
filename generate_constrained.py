import logging
import torch

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from generate_token_mask import auth_ids


tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

auth_ids = torch.as_tensor(auth_ids)

torch.set_num_threads(2)
logger = logging.getLogger(__name__)

def generate_constrained(input_sentence, max_new_tokens=10, temperature=0.5, best=True):
    past = None
    enc = tok(input_sentence, return_tensors="pt")
    ids = enc.input_ids
    nb_new_tokens = 0
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(
                input_ids=ids, 
                attention_mask=enc.attention_mask, 
                max_new_tokens=1, 
                pad_token_id=tok.eos_token_id, 
                do_sample=False,
                use_cache=True,
                past_key_values=past
            )
        past = out.past_key_values

        logits = out.logits[:, -1, :]
        auth_logits = logits[:, auth_ids]
        auth_prob = torch.softmax(auth_logits / temperature, dim=-1)
        if best:
            best = torch.argmax(auth_logits, dim=-1)
            next_token_id = auth_ids[best].unsqueeze(-1)
        else:
            sampled_best = torch.multinomial(auth_prob, num_samples=1)    
            next_token_id = auth_ids[sampled_best]
        ids = torch.cat([ids, next_token_id], dim=-1)
        logger.info(f"Generated token: {tok.decode(next_token_id[0])}")
        nb_new_tokens += 1
        if next_token_id.item() == tok.eos_token_id:
            break
    return tok.decode(ids[0]), nb_new_tokens
