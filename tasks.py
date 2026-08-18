from celery import Celery
import logging
from generate_constrained import generate_constrained, load_model
from celery.signals import worker_process_init

_model = None

logger = logging.getLogger(__name__)

app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1')

@worker_process_init.connect
def init_worker(**kwargs):
    logger.info("Initializing worker process...")
    load_model()
    logger.info("Model loaded successfully.")

    
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
