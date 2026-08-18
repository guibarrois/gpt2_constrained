from celery import Celery
from generate_constrained import generate_constrained
from celery.signals import worker_process_init

_model = None

app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1')

@app.task(bind=True)
def generate(self, input_sentence, max_new_tokens=10, temperature=0.5, best=True):
    completed_text, nb_new_tokens = generate_constrained(
        input_sentence, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature,
        best=best
    )
    return {
        "completed_text": completed_text,
        "nb_new_tokens": nb_new_tokens
    }
