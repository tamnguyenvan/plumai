import os
import base64
import uuid
import webbrowser

from flask import Flask, request, jsonify, render_template
import requests


tasks = {}

app = Flask(__name__)


@app.route('/')
def index():
    model_name = os.getenv("MODEL_NAME")
    return render_template('index.html', model_name=model_name)


@app.route('/api/generate-image', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"status": "failed", "message": "Missing prompt"}), 400

        try:
            params = validate_arguments(data)
        except ValueError as e:
            return jsonify({"status": "failed", "message": f"Invalid params: {data}"}), 400

        print(f"[API] Generating image: {params}")
        task_id = str(uuid.uuid4())
        tasks[task_id] = {"status": "processing"}
        endpoint = os.getenv("MODEL_ENDPOINT")

        # Use a background thread or async task to handle image generation
        generate_image(task_id, params, endpoint)
        return jsonify({"status": "processing", "task_id": task_id}), 202
    except Exception as e:
        return jsonify({"status": "failed", "message": f"Error: {str(e)}"}), 500


@app.route('/api/task-status/<task_id>', methods=['GET'])
def task_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"status": "failed", "message": "Task not found"}), 404
    return jsonify(task)


def generate_image(task_id, params, endpoint):
    try:
        response = requests.get(endpoint, params=params, timeout=300)
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            if "image" in content_type:
                image_bytes = response.content
                image_base64 = base64.b64encode(image_bytes).decode()
                tasks[task_id] = {
                    "status": "completed",
                    "image_base64": image_base64
                }
            else:
                tasks[task_id] = {
                    "status": "failed",
                    "message": "Unexpected content type"
                }
        else:
            tasks[task_id] = {
                "status": "failed",
                "message": f"Unexpected status code: {response.status_code}"
            }
    except Exception as e:
        tasks[task_id] = {
            "status": "failed",
            "message": f"Error: {str(e)}"
        }

def validate_and_convert(value, expected_type, default, min_value=None, max_value=None):
    """Convert and validate a value."""
    try:
        if expected_type == int:
            value = int(value)
        elif expected_type == float:
            value = float(value)
        else:
            raise ValueError("Unsupported type for validation")

        if min_value is not None and value < min_value:
            raise ValueError(f"Value must be greater than or equal to {min_value}.")
        if max_value is not None and value > max_value:
            raise ValueError(f"Value must be less than or equal to {max_value}.")

    except (ValueError, TypeError):
        return default

    return value


def validate_arguments(data):
    # Get and validate the values from the data with default values
    width = validate_and_convert(data.get("width", 1024), int, 1024, min_value=1)
    height = validate_and_convert(data.get("height", 1024), int, 1024, min_value=1)
    seed = validate_and_convert(data.get("seed", -1), int, -1)
    guidance_scale = validate_and_convert(data.get("guidance_scale", 3.5), float, 3.5, min_value=0)
    n_steps = validate_and_convert(data.get("n_steps", 20), int, 20, min_value=1)

    # Return validated arguments
    return {
        "prompt": data["prompt"],
        "width": width,
        "height": height,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "n_steps": n_steps
    }


if __name__ == '__main__':
    port = 6868
    webbrowser.open(f"http://localhost:{port}")
    app.run(host='0.0.0.0', port=port)