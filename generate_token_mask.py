from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import random
import torch
tok = GPT2TokenizerFast.from_pretrained("gpt2")

vocab = tok.get_vocab()
auth_ids = [
    token_id for token_id in vocab.values() 
    if not any([x in tok.decode(token_id) for x in ["c", "C", "ç", "Ç"]])
]

mask = torch.full((len(vocab),), float("-inf"))
mask[auth_ids] = 0

k = 20
unauth_ids = list(set(vocab.values()) - set(auth_ids))

print("Sample of authorized token ids:", [tok.decode(token_id) for token_id in random.sample(auth_ids, k=k)])
print("Sample of tokens:", [tok.decode(token_id) for token_id in random.sample(list(vocab.values()), k=k)])
print("Sample of unauthorized token ids:", [tok.decode(token_id) for token_id in random.sample(unauth_ids, k=k)])