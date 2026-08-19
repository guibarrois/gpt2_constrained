from transformers import PreTrainedTokenizerFast
import torch


def generate_mask(tok: PreTrainedTokenizerFast):

    vocab = tok.get_vocab()
    auth_ids = [
        token_id for token_id in vocab.values() 
        if not any([x in tok.decode(token_id) for x in ["c", "C", "ç", "Ç"]])
    ]
    mask = torch.full((len(vocab),), float("-inf"))
    mask[auth_ids] = 0
    return mask
