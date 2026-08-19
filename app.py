from flask import Flask, request
from tasks import generate
import time
from generate_constrained import generate_constrained

app = Flask(__name__)

@app.post("/generate")
def generate_text():
    """Endpoint to generate text based on the input sentence.

    This endpoint creates a Celery task to complete the `text`. It
    returns the `id` of the celery task, and the task is executed 
    asynchronously.
    The client can then use the `/result/<task_id>` endpoint to check the status of the task and retrieve the result once it's completed.

    The json of the request should contain at least the `text` key, and can optionally contain `max_token`, `temperature`, and `best` keys to customize the generation.
    """
    
    data = request.get_json()
    r = generate.delay(
        input_sentence=data["text"],
        max_new_tokens=data.get("max_token", 20),
        temperature=data.get("temperature", 0.5),
        best=data.get("best", True)
    )
    return {"task_id": r.id}, 202

@app.get("/result/<task_id>")
def get_result(task_id):
    """Retrieve the result of a generation task.

    The ID of the task is the one returned by the `/generate` endpoint. This endpoint checks the status of the task and returns the result if it's completed.
    
    task_id (`str`): The ID of the Celery task.
    Returns:
        dict: A dictionary containing the status of the task and, if completed, the result.
    """
    r = generate.AsyncResult(task_id)
    if not r.ready():
        return {"status": "pending"}, 202
    else:
        return {"status": "done", "result": r.get()}, 200
        

@app.post("/complete")
def complete_text():
    """Simple completion endpoint."""
    data = request.get_json()
    text = data["text"]
    app.logger.info(f"Received text for completion: {text}")
    if "best" in data:
        best = data["best"]
    else:
        best = True  # Default value if not provided
    if "max_token" in data:
        max_token = data["max_token"]
    else:
        max_token = 20  # Default value if not provided
    if "temperature" in data:
        temperature = data["temperature"]
    else:  
        temperature = 0.5  # Default value if not provided
    app.logger.info(f"Using max_token: {max_token} and temperature: {temperature}")

    t1 = time.time()
    completed_text, nb_new_tokens = generate_constrained(
        text, 
        max_new_tokens=max_token, 
        temperature=temperature,
        best=best
    )
    t2 = time.time()
    app.logger.info(f"Generation took {nb_new_tokens/ (t2-t1):.2f} t/s.")
    return {"completed_text": completed_text}

if __name__ == "__main__":
    app.run(debug=False)