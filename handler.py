import os
import runpod
import time  

import llama_cpp


MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Qwen/Qwen3-0.6B")
GGUF_NAME = os.environ.get("GGUF_NAME", "Qwen3.5-0.8B-UD-IQ2_M.gguf")

GGUF_PATH = os.path.join(MODEL_PATH, GGUF_NAME)

if not os.path.exists(GGUF_PATH):
    raise FileNotFoundError(
        f"Model not found at {GGUF_PATH}. "
        f"Please mount the model volume or set MODEL_PATH and GGUF_NAME env vars correctly."
    )

model = llama_cpp.Llama(model_path=GGUF_PATH)

def handler(event):
    print(f"Worker Start")
    input = event['input']
    response = model.create_chat_completion(
        messages=[{
            "role": "user",
            "content": input.get("prompt", "Hellp")
        }]
    )

    return response

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
