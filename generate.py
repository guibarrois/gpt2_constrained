import sys, traceback

print("starting generate.py", flush=True)


import time
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

enc = tok(
    """The French capital is""",
    return_tensors="pt"
)
ids = enc.input_ids

t0 = time.time()
out = model.generate(
    input_ids=ids,
    attention_mask=enc.attention_mask,
    max_new_tokens=20,
    pad_token_id=tok.eos_token_id,
    do_sample=False,
)
dt = time.time() - t0

print(tok.decode(out[0]))
print(f"{20/dt:.1f} tok/s")