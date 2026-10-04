"""Checked local launch, readiness and browser opening; no installs/downloads."""

import argparse
import importlib
import json
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parent


def read_status(url):
    with urlopen(url + "api/status", timeout=2) as response:
        return json.load(response)


def preflight(config_path):
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use the existing Python 3.11 .venv, not another Python installation")
    for package in ("torch", "torchvision", "cv2", "fastapi", "uvicorn", "multipart", "imageio_ffmpeg"):
        try:
            importlib.import_module(package)
        except ImportError as error:
            raise RuntimeError(f"Missing dependency {package}; restore the environment with uv sync --frozen") from error
    from dashboard_service import load_config
    from video_inference import file_sha256
    import imageio_ffmpeg
    config = load_config(config_path)
    checkpoint = (ROOT / config["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"Missing trusted model artifact: {checkpoint}. See README.md; launcher never downloads weights")
    if file_sha256(checkpoint) != config["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash mismatch; do not substitute another model")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-version"], check=True, capture_output=True, timeout=10)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=Path, default=ROOT / "dashboard_config.json")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="preflight only")
    args = parser.parse_args()
    process = None
    try:
        if not 1 <= args.port <= 65535:
            raise RuntimeError("Port must be in 1..65535")
        config = preflight(args.config.resolve())
        print("Preflight passed: environment, configuration, checkpoint SHA-256, FFmpeg", flush=True)
        if args.check:
            return 0
        url = f"http://127.0.0.1:{args.port}/"
        with socket.socket() as probe:
            occupied = probe.connect_ex(("127.0.0.1", args.port)) == 0
        if occupied:
            try:
                state = read_status(url)
                matches = state.get("model", {}).get("checkpoint_sha256") == config["checkpoint_sha256"]
            except (URLError, ValueError, AttributeError, TimeoutError):
                matches = False
            if not matches or state.get("model_state") != "ready":
                raise RuntimeError(f"Port {args.port} is occupied by another/not-ready service; stop it or use --port 8001")
            print(f"Existing ready dashboard reused: {url}. No duplicate process started.", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            return 0
        process = subprocess.Popen([sys.executable, str(ROOT / "dashboard.py"), "--port", str(args.port),
                                    "--config", str(args.config.resolve())], cwd=ROOT,
                                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Backend exited ({process.returncode}); see the visible log above")
            try:
                state = read_status(url)
                if state["model_state"] == "error":
                    raise RuntimeError(f"Model initialization failed: {state['error']}")
                if state["model_state"] == "ready":
                    break
            except (URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(.25)
        else:
            raise RuntimeError("Model readiness timed out after 120 seconds")
        print(f"READY: {url} | {state['model']['device_name']} | Ctrl+C to stop. Closing the backend stops the site.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return process.wait()
    except KeyboardInterrupt:
        print("Stopping dashboard...", flush=True)
        return 0
    except Exception as error:
        print(f"LAUNCH FAILED: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if process and process.poll() is None:
            # Uvicorn handles Ctrl+Break on Windows. Give it time to release its
            # owned temporary copies before resorting to forced termination.
            try:
                if sys.platform == "win32":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.terminate()
                process.wait(timeout=8)
            except (subprocess.TimeoutExpired, OSError):
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
