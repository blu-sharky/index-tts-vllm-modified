import os
import asyncio
import io
import tempfile
import traceback
from fastapi import FastAPI, Request, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlparse
import aiohttp
import uvicorn
import argparse
import json
import time
import soundfile as sf
from typing import List, Optional, Union

from loguru import logger
logger.add("logs/api_server_v2.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

from indextts.infer_vllm_v2 import IndexTTS2

tts = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    yield


app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "TTS model not initialized"
            }
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Service is running",
            "timestamp": time.time()
        }
    )


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


async def _download_to_temp(url: str) -> str:
    """Download a URL to a temporary file, return its path."""
    parsed = urlparse(url)
    suffix = os.path.splitext(parsed.path)[1] or ".wav"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                tmp.write(await resp.read())
            finally:
                tmp.close()
            return tmp.name


@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    temp_files = []
    try:
        data = await request.json()
        emo_control_method = data.get("emo_control_method", 0)
        text = data["text"]
        spk_audio_path = data["spk_audio_path"]
        emo_ref_path = data.get("emo_ref_path", None)
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0] * 8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)

        # Download remote files if URLs are provided
        if _is_url(spk_audio_path):
            spk_audio_path = await _download_to_temp(spk_audio_path)
            temp_files.append(spk_audio_path)
        if emo_ref_path and _is_url(emo_ref_path):
            emo_ref_path = await _download_to_temp(emo_ref_path)
            temp_files.append(emo_ref_path)

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 3:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "error": "emo_control_method=3 (text emotion) is not supported. "
                             "QwenEmotion model is disabled. Use methods 0, 1, or 2."
                }
            )
        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        if emo_control_method == 1:
            emo_weight = emo_weight
        if emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error": "情感向量之和不能超过1.5，请调整后重试。"
                    }
                )
        else:
            vec = None

        sr, wav = await tts.infer(spk_audio_prompt=spk_audio_path, text=text,
                        output_path=None,
                        emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                        emo_vector=vec,
                        use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                        max_text_tokens_per_sentence=int(max_text_tokens_per_sentence))

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )
    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except OSError:
                pass


@app.post("/tts", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_multipart(
    text: str = Form(...),
    spk_audio: UploadFile = File(...),
    emo_control_method: int = Form(0),
    emo_weight: float = Form(1.0),
):
    """multipart/form-data 上传参考音频合成语音（供无共享文件系统的客户端使用）。"""
    temp_files = []
    try:
        suffix = os.path.splitext(spk_audio.filename)[1] or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(await spk_audio.read())
        finally:
            tmp.close()
        spk_audio_path = tmp.name
        temp_files.append(spk_audio_path)

        if emo_control_method == 3:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "error": "emo_control_method=3 (text emotion) is not supported. "
                             "QwenEmotion model is disabled. Use methods 0, 1, or 2."
                }
            )

        global tts
        vec = None
        if emo_control_method == 0:
            emo_weight = 1.0

        sr, wav = await tts.infer(
            spk_audio_prompt=spk_audio_path, text=text,
            output_path=None,
            emo_audio_prompt=None, emo_alpha=emo_weight,
            emo_vector=vec,
            use_emo_text=False, emo_text=None, use_random=False,
            max_text_tokens_per_sentence=120,
        )

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(tb_str)}
        )
    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except OSError:
                pass



    # 从 backend/config.json 读取 TTS 配置作为默认值
    def _load_config_defaults() -> dict:
        here = os.path.dirname(os.path.abspath(__file__))
        for _ in range(6):
            candidate = os.path.join(here, "config.json")
            if os.path.exists(candidate):
                import json
                with open(candidate, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                yt = raw.get("youtube", {})
                return {
                    "model_dir": yt.get("tts_model_dir", "checkpoints/IndexTTS-2-vLLM"),
                    "gpu_memory_utilization": float(yt.get("tts_gpu_mem", 0.25)),
                    "port": int(yt.get("tts_port", 6006)),
                }
            here = os.path.dirname(here)
        return {}

    _cfg = _load_config_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=_cfg.get("port", 6006))
    parser.add_argument("--model_dir", type=str, default=_cfg.get("model_dir", "checkpoints/IndexTTS-2-vLLM"), help="Model checkpoints directory")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
    parser.add_argument("--gpu_memory_utilization", type=float, default=_cfg.get("gpu_memory_utilization", 0.25))
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    args = parser.parse_args()

    logger.info(f">> Config: model_dir={args.model_dir}, port={args.port}, gpu_mem={args.gpu_memory_utilization}")
    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    uvicorn.run(app=app, host=args.host, port=args.port)