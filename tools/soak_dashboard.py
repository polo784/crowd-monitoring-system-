"""Explicit real-backend soak, not a browser playback test. No training.

Run with an already-ready dashboard. Creates only ignored diagnostic outputs.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import cv2
import httpx
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--backend-pid", type=int, required=True)
    args = parser.parse_args()
    out = ROOT / "outputs/dashboard/release_soak"
    out.mkdir(parents=True, exist_ok=True)
    clip = out / "repeated_diagnostic.mp4"
    # Stream-copy the existing short diagnostic into a long source. It is then
    # converted by the real upload endpoint, exactly like any unsupported codec.
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-stream_loop", "-1", "-i", str(ROOT / "videos/ht21/HT21-01-raw.mp4"),
                    "-t", str(args.seconds + 30), "-c", "copy", str(clip)], check=True, timeout=60)
    client = httpx.Client(base_url=args.url, timeout=660)
    state = client.get("/api/status").raise_for_status().json()
    assert state["model_state"] == "ready", state
    with clip.open("rb") as stream:
        session = client.post("/api/video", files={"file": (clip.name, stream, "video/mp4")}).raise_for_status().json()
    playback = out / "playback.mp4"
    with client.stream("GET", session["video_url"]) as response:
        response.raise_for_status()
        with playback.open("wb") as stream:
            for block in response.iter_bytes(): stream.write(block)
    client.post("/api/control", json={"session_id": session["id"], "generation": 1,
                "action": "play", "timestamp": 0, "playing": True}).raise_for_status()
    cap = cv2.VideoCapture(str(playback))
    samples, health, resources, failures = [], [], [], []
    start, next_sample, next_health = time.perf_counter(), 0., 0.
    try:
        while (elapsed := time.perf_counter() - start) < args.seconds:
            if elapsed < next_sample:
                time.sleep(min(.05, next_sample-elapsed)); continue
            next_sample = elapsed + .25
            cap.set(cv2.CAP_PROP_POS_MSEC, elapsed * 1000)
            ok, frame = cap.read()
            if not ok: raise RuntimeError("Diagnostic playback decode failed")
            captured = time.perf_counter()
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            assert ok
            response = client.post("/api/infer", data={"session_id": session["id"], "generation": 1,
                         "timestamp": elapsed, "capture_time": datetime.now(timezone.utc).isoformat()},
                         files={"frame": ("frame.jpg", jpeg.tobytes(), "image/jpeg")}, timeout=30)
            if response.status_code != 200:
                failures.append({"time": elapsed, "status": response.status_code, "body": response.text}); break
            result = response.json()
            samples.append({k: v for k, v in result.items() if k != "heatmap"} | {"diagnostic_client_latency_ms": (time.perf_counter()-captured)*1000})
            if elapsed >= next_health:
                before = time.perf_counter()
                current = client.get("/api/status", timeout=5).raise_for_status().json()
                health.append({"time": elapsed, "ms": (time.perf_counter()-before)*1000, "ready": current["model_state"]})
                # Windows process/GPU memory sampled by normal read-only CLIs.
                ps = subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"Get-Process -Id {args.backend_pid} | Select-Object Id,WorkingSet64,PrivateMemorySize64,HandleCount | ConvertTo-Json -Compress"], capture_output=True, text=True, timeout=10)
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                resources.append({"time": elapsed, "process_memory": ps.stdout.strip(), "gpu_total_used_mib": gpu.stdout.strip()})
                print(f"Soak {elapsed:.0f}/{args.seconds}s | {len(samples)} real GPU samples | health {health[-1]['ms']:.1f} ms", flush=True)
                next_health = elapsed + 30
    finally:
        cap.release()
    retained = client.get("/api/export").raise_for_status().json()
    total_seconds = time.perf_counter() - start
    result = {"scope": "real HTTP upload/conversion/JPEG/GPU/backend soak; NOT browser playback", "seconds": total_seconds,
              "samples": len(samples), "failures": failures, "model": state["model"], "config": state["config"],
              "video": session["video"], "health": health, "resources": resources,
              "mean_forward_ms": float(np.mean([s["inference_ms"] for s in samples])),
              "mean_service_ms": float(np.mean([s["service_ms"] for s in samples])),
              "mean_diagnostic_client_latency_ms": float(np.mean([s["diagnostic_client_latency_ms"] for s in samples])),
              "effective_samples_per_second": len(samples)/total_seconds,
              "input_shape": samples[-1]["input_shape"], "output_shape": samples[-1]["output_shape"],
              "retained_count": len(retained["observations"]), "retained_metadata": retained["metadata"]}
    assert len(retained["observations"]) <= state["config"]["history_limit"]
    assert retained["metadata"]["observations_total"] == len(samples)
    (out / "soak_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "retained_measurements.json").write_text(json.dumps(retained, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in {"resources", "health", "video", "model", "config", "retained_metadata"}}, indent=2), flush=True)
    client.post("/api/control", json={"session_id": session["id"], "generation": 2,
                "action": "pause", "timestamp": min(args.seconds, session["video"]["duration"]), "playing": False})
    client.close()
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
