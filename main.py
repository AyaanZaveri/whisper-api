import gradio_client as grc
from fastapi import FastAPI

app = FastAPI()

generator_client = grc.Client("ayaanzaveri/faster-whisper-api-main", hf_token="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")

@app.get("/predict")
async def predict(url: str, word_timestamps: str = "false", model: str = "tiny"):
    job = generator_client.predict(url, word_timestamps, model)
    while not job.done():
        pass

    return job.result()
