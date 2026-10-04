"""Real checkpoint/count invariance and explicit input-budget diagnostic (no training)."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import torch
from crowd_inference import CSRNetPredictor
from dashboard_service import compatible_size, load_config
from video_inference import file_sha256


def main():
    config = load_config()
    assert torch.cuda.is_available(), "This diagnostic requires the laptop CUDA GPU"
    checkpoint = ROOT / config["checkpoint"]
    assert file_sha256(checkpoint) == config["checkpoint_sha256"]
    predictor = CSRNetPredictor(checkpoint, device="cuda")
    report = {"model": predictor.metadata, "checkpoint_sha256": file_sha256(checkpoint), "samples": []}
    for source in ("videos/ht21/HT21-01-raw.mp4", "videos/video1.mp4"):
        cap = cv2.VideoCapture(str(ROOT / source))
        ok, frame = cap.read(); cap.release(); assert ok
        original_hw = list(frame.shape[:2])
        h, w = compatible_size(*frame.shape[:2], config["max_model_pixels"])
        prepared = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        torch.cuda.reset_peak_memory_stats()
        predictions = [predictor.predict(prepared) for _ in range(3)]
        result = {"source": source, "original_hw": original_hw, "pixel_budget": config["max_model_pixels"],
                  "model_hw": [h,w], "input_shape": predictions[-1].input_shape,
                  "output_shape": predictions[-1].output_shape, "raw_count": predictions[-1].count,
                  "forward_ms": [p.inference_ms for p in predictions],
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
        if "ht21" in source:
            previous = 35.3098713690415  # unchanged Oct4 audit of same first native frame/size
            result["previous_audit_raw_count"] = previous
            result["raw_count_difference"] = predictions[-1].count - previous
            assert abs(result["raw_count_difference"]) < 1e-6, result
        report["samples"].append(result)
        print(json.dumps(result), flush=True)
    out = ROOT / "outputs/dashboard/release_soak/gpu_budget.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert file_sha256(checkpoint) == config["checkpoint_sha256"]


if __name__ == "__main__": main()
