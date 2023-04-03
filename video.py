import pathlib

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
    }
    try:
        print(f"Downloading video from {video_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(video_url)
        print(f"Downloaded video from {video_url} to {destination_path}")
    except Exception as e:
        raise (e)