import os
import runpod
import time  

import llama_cpp


MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Qwen/Qwen3-0.6B")
model = llama_cpp.Llama(model_path=MODEL_PATH)

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
