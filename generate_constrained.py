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
    """Load the GPT-2 model and tokenizer, and generate the token mask.
    
    The model and tokenizer are loaded from the pre-trained GPT-2 model. The token mask is generated to restrict the output tokens based on specific criteria.
    """
    global _model, _tok, _mask
    _tok = GPT2TokenizerFast.from_pretrained("gpt2")
    _model = GPT2LMHeadModel.from_pretrained("gpt2")
    _mask = generate_mask(_tok)
    _model.eval()

def generate_constrained(input_sentence, max_new_tokens=10, temperature=0.5, best=True):
    """Generate text based on the input sentence with constraints.

    Generation is done token by token, and the mask is apply to the logits to restrict the output tokens. The token is then chosen according to the
    strategy defined by the `best` parameter (argmax or sampling).
    
    input_sentence (`str`): The input text to generate from.
    max_new_tokens (`int`, *optional*, defaults to 10): The maximum number of new tokens to generate.
    temperature (`float`, *optional*, defaults to 0.5): The temperature for sampling. Higher values lead to more random outputs.
    best (`bool`, *optional*, defaults to True): Whether to use the best token (argmax) or sample from the distribution.
    Returns:
        tuple: A tuple containing the completed text and the number of new tokens generated.
    """
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
