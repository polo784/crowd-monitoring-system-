"""Start the local single-video crowd dashboard: python dashboard.py."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import logging
import math
from pathlib import Path
import subprocess
import tempfile
import copy
from typing import Literal

import cv2
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dashboard_service import DashboardRuntime, ROOT, RuntimeProblem, compatible_size, load_config

LOG = logging.getLogger("crowd.dashboard")


def inspect_video(path, config):
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or not math.isfinite(fps) or fps <= 0 or frames <= 0:
            raise RuntimeProblem(422, "Cannot decode this video or determine its duration")
        h, w = frame.shape[:2]
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        return {"width": w, "height": h, "fps": fps, "reported_frames": frames,
                "duration": frames / fps, "codec": "".join(chr((codec >> (8*i)) & 255) for i in range(4)),
                "model_size_hw": compatible_size(h, w, config["max_model_pixels"])}
    finally:
        cap.release()


def prepare_playback(path, scratch, metadata, config):
    # Browsers cannot reliably play OpenCV's mp4v outputs. Convert only the copy,
    # not the user's source. Common H.264 MP4s pass through without re-encoding.
    metadata["original"] = copy.deepcopy(metadata)
    if path.suffix.lower() == ".mp4" and metadata["codec"].lower() in {"h264", "avc1"}:
        metadata["playback_conversion"] = False
        metadata["playback"] = copy.deepcopy(metadata["original"])
        metadata["conversion"] = {"performed": False, "audio_removed": False, "padding_hw": [0, 0]}
        return path
    import imageio_ffmpeg
    converted = Path(scratch.name) / "playback.mp4"
    LOG.info("Preparing H.264 browser copy for %s codec", metadata["codec"])
    process = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                              "-nostdin", "-n", "-i", str(path), "-map", "0:v:0", "-an",
                              "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                              "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                              "-movflags", "+faststart", str(converted)],
                             capture_output=True, timeout=600)
    if process.returncode:
        raise RuntimeProblem(422, "Could not prepare browser playback. Try an H.264 MP4 video.")
    converted_metadata = inspect_video(converted, config)
    # The browser's capture dimensions must describe the playback copy.
    metadata.update(converted_metadata)
    metadata["playback_conversion"] = True
    metadata["playback"] = converted_metadata
    metadata["conversion"] = {"performed": True, "codec": "libx264", "preset": "fast", "crf": 18,
                              "pixel_format": "yuv420p", "audio_removed": True,
                              "padding_hw": [metadata["height"] - metadata["original"]["height"],
                                             metadata["width"] - metadata["original"]["width"]]}
    return converted


class Control(BaseModel):
    session_id: str
    generation: int = Field(ge=1)
    action: Literal["play", "pause", "seek", "restart", "ended", "invalidate", "zones"]
    timestamp: float = Field(ge=0, allow_inf_nan=False)
    playing: bool
    zones: list[dict] | None = Field(default=None, max_length=6)


def create_app(config=None, runtime=None):
    config = config or load_config()
    runtime = runtime or DashboardRuntime(config)

    @asynccontextmanager
    async def lifespan(app):
        loading = asyncio.create_task(asyncio.to_thread(runtime.load_model)) if runtime.predictor is None and runtime.model_state == "loading" else None
        yield
        if loading:
            await loading
        runtime.close()

    app = FastAPI(title="Local Crowd Intelligence", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.runtime = runtime
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method != "GET" and origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Use the dashboard from its own local URL"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(RuntimeProblem)
    async def runtime_problem(request, error):
        return JSONResponse({"detail": error.message}, status_code=error.status)

    @app.get("/")
    def index():
        return FileResponse(ROOT / "dashboard" / "index.html")

    @app.get("/api/status")
    def status():
        return runtime.status()

    @app.get("/api/export")
    def export():
        return runtime.export_history()

    @app.post("/api/video")
    async def upload_video(file: UploadFile = File(...)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}:
            raise RuntimeProblem(422, "Choose an MP4, MOV, AVI, MKV or WebM video")
        scratch = tempfile.TemporaryDirectory(prefix="crowd-video-", ignore_cleanup_errors=True)
        original = Path(scratch.name) / ("source" + suffix)
        try:
            size = 0
            with original.open("xb") as stream:
                while block := await file.read(1024 * 1024):
                    size += len(block)
                    if size > config["max_video_bytes"]:
                        raise RuntimeProblem(413, "Video exceeds the configured local upload limit")
                    stream.write(block)
            metadata = await asyncio.to_thread(inspect_video, original, config)
            path = await asyncio.to_thread(prepare_playback, original, scratch, metadata, config)
            name = (file.filename or "video").replace("\\", "/").split("/")[-1]
            session = runtime.select_video(path, name, metadata, scratch)
            session["video_url"] = f"/api/video/{session['id']}"
            session["export_metadata"] = runtime.export_metadata()
            return session
        except RuntimeProblem:
            scratch.cleanup()
            raise
        except Exception as error:
            scratch.cleanup()
            LOG.exception("Video preparation failed")
            raise RuntimeProblem(422, "Video preparation failed; try a smaller H.264 MP4") from error
        finally:
            await file.close()

    @app.get("/api/video/{session_id}")
    def video_file(session_id: str):
        with runtime.lock:
            path = runtime.require_session(session_id).path
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "no-store"})

    @app.post("/api/control")
    def control(command: Control):
        return runtime.control(**command.model_dump())

    @app.post("/api/infer")
    async def infer(session_id: str = Form(...), generation: int = Form(...), timestamp: float = Form(...),
                    snapshot: bool = Form(False), capture_time: str | None = Form(None), frame: UploadFile = File(...)):
        try:
            data = await frame.read(config["max_frame_bytes"] + 1)
            return await asyncio.to_thread(runtime.infer, session_id, generation, timestamp, data, snapshot, capture_time)
        except RuntimeProblem:
            raise
        except Exception as error:
            LOG.exception("Inference failed")
            raise RuntimeProblem(500, "Inference failed; current measurements are unavailable. Check the backend log before retrying.") from error
        finally:
            await frame.close()

    app.mount("/static", StaticFiles(directory=ROOT / "dashboard"), name="static")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=Path, default=ROOT / "dashboard_config.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import uvicorn
    LOG.info("Open http://127.0.0.1:%s | one video, one model, local only", args.port)
    uvicorn.run(create_app(load_config(args.config)), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
