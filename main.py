import asyncio
import base64
import io
import os
import time
from pathlib import Path

import edge_tts
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai
from google.genai import types

app = FastAPI(title="Ayzo.ai")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).resolve().parent
MALE_VOICE = "en-US-AndrewNeural"

SYSTEM_PROMPT = """You are Ayzo.ai, a futuristic, calm, and highly intelligent AI assistant.
Keep your answers clear, structured, and natural to both read on screen and hear spoken aloud."""

def get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not set on the server.")
    return genai.Client(api_key=api_key)

def generate_with_retry(client: genai.Client, contents, system_instruction: str | None = None, temperature: float = 0.7) -> str:
    config = types.GenerateContentConfig(temperature=temperature)
    if system_instruction:
        config.system_instruction = system_instruction

    models_to_try = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
    last_error = None
    for m in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(model=m, contents=contents, config=config)
                return (response.text or "").strip()
            except Exception as e:
                last_error = e
                if any(c in str(e) for c in ["503", "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]):
                    time.sleep(1)
                    continue
                raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")
    raise HTTPException(status_code=502, detail=f"All models are temporarily busy: {last_error}")

async def synthesize_male_speech_base64(text: str) -> str:
    clean_text = text.replace("**", "").replace("__", "").replace("`", "").replace("#", "")
    communicate = edge_tts.Communicate(clean_text, voice=MALE_VOICE)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

@app.get("/")
def serve_home():
    return FileResponse(BASE_DIR / "index.html")

@app.post("/api/chat")
async def chat_endpoint(
    prompt: str = Form(""),
    model: str = Form("gemini-3.5-flash-lite"),
    voice_output: bool = Form(True),
    audio: UploadFile | None = File(None),
):
    client = get_client()
    user_text = prompt.strip()
    if audio is not None:
        audio_bytes = await audio.read()
        if audio_bytes:
            user_text = generate_with_retry(
                client=client,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=audio.content_type or "audio/webm"),
                    "Transcribe this spoken audio accurately into plain text. Return only the transcription.",
                ],
                temperature=0.1,
            )
    if not user_text:
        raise HTTPException(status_code=400, detail="Please speak or type a question.")
    answer_text = generate_with_retry(
        client=client,
        contents=user_text,
        system_instruction=SYSTEM_PROMPT,
        temperature=0.7,
    )
    audio_b64 = None
    if voice_output and answer_text:
        try:
            audio_b64 = await asyncio.wait_for(synthesize_male_speech_base64(answer_text), timeout=5.0)
        except Exception:
            audio_b64 = None
    return {"transcript": user_text, "answer": answer_text, "audio_base64": audio_b64, "audio_mime": "audio/mp3"}
