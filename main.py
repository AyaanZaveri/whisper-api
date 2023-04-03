from faster_whisper import WhisperModel
from fastapi import FastAPI
from video import download_convert_video_to_audio
import yt_dlp
import uuid
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8

def segment_to_dict(segment):
    segment = segment._asdict()
    if segment["words"] is not None:
        segment["words"] = [word._asdict() for word in segment["words"]]
    return segment

@app.post("/video")
async def download_video(video_url: str):
    download_convert_video_to_audio(yt_dlp, video_url, f"/home/user/{uuid.uuid4().hex}")

@app.post("/transcribe")
async def transcribe_video(video_url: str, beam_size: int = 5, model_size: str = "tiny", word_timestamps: bool = True):
    print("loading model")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("getting hex")
    rand_id = uuid.uuid4().hex
    print("doing download")
    download_convert_video_to_audio(yt_dlp, video_url, f"/home/user/{rand_id}")
    segments, info = model.transcribe(f"/home/user/{rand_id}.mp3", beam_size=beam_size, word_timestamps=word_timestamps)
    segments = [segment_to_dict(segment) for segment in segments]
    total_duration = round(info.duration, 2)  # Same precision as the Whisper timestamps.
    print(info)
    os.remove(f"/home/user/{rand_id}.mp3")
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    
    return segments

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
