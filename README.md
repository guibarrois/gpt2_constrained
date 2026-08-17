# Constrained generation

The purpose of this repository is to demonstrate constrained generation with gpt-2. It
implements a simple generation function using the transformers library, that also
applies a masking on the output token.

## Implementation

`generate_token_mask.py` extract the model vocabulary from the tokenizer, and
return the valid ids according to a simple prodicate. Currently it is very simple
(e.g. token not containing a list of given characters) and ineficient but I'll work
on that later to improve that.

`generate_constrained.py` contained the base function that does the generation. Logits
and probabilty are computed, masked, and the next character is selected according to
two possible strategies:
- `best=True` --> select the highest probability token (deterministic)
- `best=False` --> do multinomial sampling (non-deterministic, with a temperature parameter)
