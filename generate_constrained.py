import logging
import torch

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from generate_token_mask import generate_mask





torch.set_num_threads(2)
logger = logging.getLogger(__name__)

_model = None
_tok = None
_mask = None

def load_model():
    global _model, _tok, _mask
    _tok = GPT2TokenizerFast.from_pretrained("gpt2")
    _model = GPT2LMHeadModel.from_pretrained("gpt2")
    _mask = generate_mask(_tok)
    _model.eval()

def generate_constrained(input_sentence, max_new_tokens=10, temperature=0.5, best=True):
    past = None
    enc = _tok(input_sentence, return_tensors="pt")
    ids = enc.input_ids
    cur = ids
    nb_new_tokens = 0
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = _model(
                input_ids=cur,
                use_cache=True,
                past_key_values=past
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :] + _mask
            if best:
                best_token = torch.argmax(logits, dim=-1)
                next_token_id = best_token.unsqueeze(-1)
            else:
                prob = torch.softmax(logits / temperature, dim=-1)
                next_token_id = torch.multinomial(prob, num_samples=1)    
        ids = torch.cat([ids, next_token_id], dim=-1)
        cur = next_token_id
        logger.info(f"Generated token: {_tok.decode(next_token_id[0])}")
        nb_new_tokens += 1
        if next_token_id.item() == _tok.eos_token_id:
            break
    return _tok.decode(ids[0]), nb_new_tokens
