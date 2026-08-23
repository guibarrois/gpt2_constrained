# Quick start

This repository aims at demonstrating two, things:
- constrained generation of a self-served model
- self hosting and scaling using Kubernetes and Celery

To run, the repository requires three components: a redis
server, a celery worker and an api. There are three
ways to run it (Docker,  Manually, Kubernetes). The most 
intersting (and the "raison d'être" of this repository)
for me is Kubernetes, which allows fine grain
control of ressources and scaling.

## Docker

You can run the three services at once using:
```
export CELERY_BROKER_URL=redis://redis:6379/0
export CELERY_BACKEND_URL=redis://redis:6379/1
docker compose up --build
```
This will run the whole infrastructure. Thanks
to ports mapping 5000:5000 in the `compose.yml` file, you
can then test the endpoints using for instance

```
curl -i -X POST http://localhost:5000/generate -H "Content-Type: application/json" -d '{"text":"The capital of Colombia is","max_token":20,"best":true}'
```
Which returns a celery task id for instance `369a4835-ba5f-49b6-ac01-71e85d01b8db`
and then 
```
curl http://localhost:5000/result/369a4835-ba5f-49b6-ac01-71e85d01b8db
```
to retrieve the results of the task.


## Manually launching 

You can also launch manually the three services:

Redis
```
docker run -d --name local-redis -p 6379:6379 redis
```

Celery worker:
```
celery -A tasks worker
```

Flask server with
```
gunicorn --bind 0.0.0.0:5000 app:app
```

and send the same message as above to test.

## Kubernetes

TO BE DONE

# Constrained generation

We implement a simple generation function using the transformers library, that also
applies a masking on the output token. The masking is kept very simple, but could
be enriched by modifying `generate_token_mask.py` file.

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

# Serving

This repository is also used to demonstrate the scalable self hosting of an LLM,
using Kubernetes and Celery as backbones.


