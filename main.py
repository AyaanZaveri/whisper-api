import pathlib
from faster_whisper import WhisperModel
import yt_dlp
import uuid
import os
import gradio as gr


# List of all supported video sites here https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md
def download_convert_video_to_audio(
    yt_dlp,
    video_url: str,
    destination_path: pathlib.Path,
) -> None:
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {  # Extract audio using ffmpeg
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
        "outtmpl": f"{destination_path}.%(ext)s",
        "concurrent-fragments": 128
    }
    try:
        print(f"Downloading video from {video_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(video_url)
        print(f"Downloaded video from {video_url} to {destination_path}")
    except Exception as e:
        raise (e)

def segment_to_dict(segment):
    segment = segment._asdict()
    if segment["words"] is not None:
        segment["words"] = [word._asdict() for word in segment["words"]]
    return segment

def download_video(video_url: str):
    download_convert_video_to_audio(yt_dlp, video_url, f"{uuid.uuid4().hex}")

def transcribe_video(video_url: str, beam_size: int = 5, model_size: str = "tiny", word_timestamps: bool = True):
    print("loading model")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("getting hex")
    rand_id = uuid.uuid4().hex
    print("doing download")
    download_convert_video_to_audio(yt_dlp, video_url, f"{rand_id}")
    print("done download")
    print("doing transcribe")
    segments, info = model.transcribe(f"{rand_id}.mp3", beam_size=beam_size, word_timestamps=word_timestamps)
    segments = [segment_to_dict(segment) for segment in segments]
    total_duration = round(info.duration, 2)  # Same precision as the Whisper timestamps.
    print(info)
    os.remove(f"{rand_id}.mp3")
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    print(segments)
    return segments

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

demo = gr.Interface(fn=transcribe_video, inputs="text", outputs="json")

demo.launch()
