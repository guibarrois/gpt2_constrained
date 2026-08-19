from celery import Celery
import logging
from generate_constrained import generate_constrained, load_model
from celery.signals import worker_process_init

logger = logging.getLogger(__name__)

app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1')

@worker_process_init.connect
def init_worker(**kwargs):
    """Load the model when a worker process starts.
    
    This is to ensure that the model is loaded only once per worker, rather
    than at each task invocation, which would be inefficient.
    """
    logger.info("Initializing worker process...")
    load_model()
    logger.info("Model loaded successfully.")


@app.task(bind=True)
def generate(self, input_sentence, max_new_tokens=10, temperature=0.5, best=True):
    """Celery task to generate text based on the input sentence.
    
    input_sentence (`str`): The input text to generate from.
    max_new_tokens (`int`, *optional*, defaults to 10): The maximum number of new tokens to generate.
    temperature (`float`, *optional*, defaults to 0.5): The temperature for sampling. Higher values lead to more random outputs.
    best (`bool`, *optional*, defaults to True): Whether to use the best token (argmax) or sample from the distribution.
    Returns:
        dict: A dictionary containing the completed text and the number of new tokens generated.
    """
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
