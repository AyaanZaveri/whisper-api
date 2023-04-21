import gradio_client as grc
from fastapi import FastAPI
import json
from pathlib2 import Path

app = FastAPI()

generator_client = grc.Client("ayaanzaveri/faster-whisper-api-main", hf_token="")

@app.get("/predict")
async def predict(url: str, word_timestamps: str = "false", model: str = "tiny"):
    job = generator_client.predict(url, word_timestamps, model)
    while not job.done():
        pass

    contents = Path(job.result()).read_text()
    return json.loads(contents)
