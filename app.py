from flask import Flask, request
from tasks import generate
import time
from generate_constrained import generate_constrained

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/generate", methods=["POST"])
def generate_text():
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
    r = generate.AsyncResult(task_id)
    if not r.ready():
        return {"status": "pending"}, 202
    else:
        return {"status": "done", "result": r.get()}, 200
        

@app.route("/complete", methods=["POST"])
def complete_text():
    # Placeholder for the completion logic
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