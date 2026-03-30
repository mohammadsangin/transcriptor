import gradio as gr
import whisper
import yt_dlp
import os
import re
import tempfile
from pathlib import Path
import imageio_ffmpeg

os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def format_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02}:{s:02}"
    return f"{m:02}:{s:02}"

def to_srt(segments):
    out = []
    for i, seg in enumerate(segments, 1):
        out.append(f"{i}\n{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n{seg['text'].strip()}\n")
    return "\n".join(out)

def to_timestamped(segments):
    return "\n".join(f"[{format_ts(seg['start'])}]  {seg['text'].strip()}" for seg in segments)

def safe_name(s):
    return re.sub(r"[^\w\-]", "_", s)[:60] or "transcript"

def save_all(result, name):
    out = Path("transcripts")
    out.mkdir(exist_ok=True)
    n = safe_name(name)
    p = out / f"{n}.txt"
    ts = out / f"{n}_timestamped.txt"
    sr = out / f"{n}.srt"
    p.write_text(result["text"].strip(), encoding="utf-8")
    ts.write_text(to_timestamped(result["segments"]), encoding="utf-8")
    sr.write_text(to_srt(result["segments"]), encoding="utf-8")
    return str(p), str(ts), str(sr)

def normalise_url(url):
    m = re.search(r"wvideo=([a-zA-Z0-9]+)", url)
    if m:
        return f"https://fast.wistia.net/embed/iframe/{m.group(1)}"
    return url

_cache = {}

def get_model(size):
    if size not in _cache:
        _cache[size] = whisper.load_model(size)
    return _cache[size]

def transcribe(audio_path, model_size, language):
    model = get_model(model_size)
    lang = None if language == "Auto-detect" else language[:2].lower()
    return model.transcribe(audio_path, language=lang, verbose=False)

def download_audio(url):
    tmp = tempfile.mkdtemp()
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmp, "%(title)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "ffmpeg_location": os.path.dirname(ffmpeg_path),
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "video")
    for f in Path(tmp).glob("*.mp3"):
        return str(f), title
    raise FileNotFoundError("Audio file not found after download.")

def handle_url(raw_url, model_size, language):
    if not raw_url.strip():
        yield "Please enter a URL.", "", "", None, None, None
        return
    url = normalise_url(raw_url.strip())
    yield "Downloading audio...", "", "", None, None, None
    try:
        audio, title = download_audio(url)
    except Exception as e:
        yield f"Download failed: {e}", "", "", None, None, None
        return
    yield f"Transcribing {title} with Whisper {model_size}...", "", "", None, None, None
    try:
        result = transcribe(audio, model_size, language)
    except Exception as e:
        yield f"Transcription error: {e}", "", "", None, None, None
        return
    p, ts, sr = save_all(result, title)
    yield (
        f"Done! {len(result['text'].split())} words transcribed.",
        Path(p).read_text(encoding="utf-8"),
        Path(ts).read_text(encoding="utf-8"),
        p, ts, sr
    )

def handle_file(file, model_size, language):
    if file is None:
        yield "Please upload a file.", "", "", None, None, None
        return
    name = Path(file.name).stem
    yield f"Transcribing {Path(file.name).name}...", "", "", None, None, None
    try:
        result = transcribe(file.name, model_size, language)
    except Exception as e:
        yield f"Transcription error: {e}", "", "", None, None, None
        return
    p, ts, sr = save_all(result, name)
    yield (
        f"Done! {len(result['text'].split())} words transcribed.",
        Path(p).read_text(encoding="utf-8"),
        Path(ts).read_text(encoding="utf-8"),
        p, ts, sr
    )

MODELS = ["tiny", "base", "small", "medium", "large"]
LANGUAGES = ["Auto-detect", "en English", "es Spanish", "fr French", "de German",
             "it Italian", "pt Portuguese", "ar Arabic", "zh Chinese", "ja Japanese", "hi Hindi"]

with gr.Blocks(title="TranscriptorAI") as demo:
    gr.Markdown("# ⚡ TranscriptorAI\nTranscribe YouTube, Wistia/Dropship Circle, or any uploaded video.")

    with gr.Row():
        model_dd = gr.Dropdown(MODELS, value="base", label="Whisper Model")
        lang_dd = gr.Dropdown(LANGUAGES, value="Auto-detect", label="Language")

    with gr.Tabs():
        with gr.Tab("🔗 URL (YouTube / Wistia / Dropship Circle)"):
            gr.Markdown("Paste a YouTube URL or right-click Dropship Circle video → Copy link and thumbnail → paste here.")
            url_in = gr.Textbox(placeholder="https://...", label="Video URL", lines=2)
            url_btn = gr.Button("⚡ Transcribe", variant="primary")
            url_status = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("Plain Text"):
                    url_plain = gr.Textbox(lines=16, label="")
                with gr.Tab("Timestamped"):
                    url_ts = gr.Textbox(lines=16, label="")
            with gr.Row():
                url_f1 = gr.File(label="Download .txt")
                url_f2 = gr.File(label="Download timestamped.txt")
                url_f3 = gr.File(label="Download .srt")
            url_btn.click(handle_url, inputs=[url_in, model_dd, lang_dd],
                outputs=[url_status, url_plain, url_ts, url_f1, url_f2, url_f3])

        with gr.Tab("📁 Upload File (mp4 / mp3 / wav...)"):
            gr.Markdown("Upload any video or audio file downloaded from Dropship Circle, Udemy, etc.")
            file_in = gr.File(label="Drop file here", file_types=[
                ".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".m4a", ".wav", ".flac", ".ogg"])
            file_btn = gr.Button("⚡ Transcribe", variant="primary")
            file_status = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("Plain Text"):
                    file_plain = gr.Textbox(lines=16, label="")
                with gr.Tab("Timestamped"):
                    file_ts = gr.Textbox(lines=16, label="")
            with gr.Row():
                file_f1 = gr.File(label="Download .txt")
                file_f2 = gr.File(label="Download timestamped.txt")
                file_f3 = gr.File(label="Download .srt")
            file_btn.click(handle_file, inputs=[file_in, model_dd, lang_dd],
                outputs=[file_status, file_plain, file_ts, file_f1, file_f2, file_f3])

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860)
