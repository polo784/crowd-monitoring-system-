"""Reusable annotated-image count benchmark. No training or count correction."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from crowd_inference import CSRNetPredictor
from dashboard_service import compatible_size
from video_inference import file_sha256


def load_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data,dict): raise ValueError("Manifest must be a JSON object")
    if data.get("schema_version") != 1 or isinstance(data.get("schema_version"), bool):
        raise ValueError("Expected count manifest schema_version 1")
    for key in ("dataset", "split", "ground_truth_source"):
        if not isinstance(data.get(key), str) or not data[key].strip(): raise ValueError(f"Manifest needs a nonempty {key}")
    if data.get("annotations_complete") is not True:
        raise ValueError("Use complete annotations; incomplete/AI draft annotations cannot establish accuracy")
    if data.get("ground_truth_kind") not in {"provided-dataset-annotations", "human-reviewed"}:
        raise ValueError("ground_truth_kind must identify provided-dataset-annotations or human-reviewed")
    samples, ids, images = data.get("samples"), set(), set()
    if not isinstance(samples, list) or not samples: raise ValueError("Manifest must contain samples")
    checked = []
    for row in samples:
        if not isinstance(row, dict): raise ValueError("Each sample must be an object")
        identity, count, image = row.get("id"), row.get("true_count"), row.get("image")
        if not isinstance(identity, str) or not identity.strip() or identity in ids: raise ValueError("Sample IDs must be nonempty and unique")
        if isinstance(count, bool) or not isinstance(count, (int, float)) or not 0 <= count <= 1000000 or not math.isfinite(count):
            raise ValueError(f"Invalid true_count for {identity}")
        if not isinstance(image, str) or not image: raise ValueError(f"Missing image path for {identity}")
        image_path = (path.parent / image).resolve()
        if not image_path.is_file(): raise FileNotFoundError(image_path)
        if image_path in images: raise ValueError("Repeated image paths would double-weight evaluation samples")
        condition = row.get("condition", "unspecified")
        if not isinstance(condition, str) or not condition.strip(): raise ValueError("condition must be a nonempty label")
        ids.add(identity); images.add(image_path)
        checked.append({"id":identity,"image":str(image_path),"true_count":float(count),"condition":condition})
    return data, checked


def count_band(count):
    if count == 0: return "empty_0"
    if count < 50: return "below_50"
    if count < 200: return "50_to_199"
    return "200_or_more"


def count_metrics(rows):
    errors = np.asarray([r["predicted_count"]-r["true_count"] for r in rows], dtype=np.float64)
    if not errors.size: return None
    return {"samples":len(rows),"mae":float(np.abs(errors).mean()),
            "rmse":float(np.sqrt(np.square(errors).mean())),"signed_bias":float(errors.mean()),
            "mean_true_count":float(np.mean([r["true_count"] for r in rows])),
            "mean_predicted_count":float(np.mean([r["predicted_count"] for r in rows]))}


def evaluate_manifest(manifest_path, predictor, max_model_pixels=921600):
    if isinstance(max_model_pixels,bool) or not isinstance(max_model_pixels,int) or not 64 <= max_model_pixels <= 2073600:
        raise ValueError("max_model_pixels must be an integer in [64, 2073600]")
    data, samples = load_manifest(manifest_path)
    rows = []
    for sample in samples:
        with Image.open(sample["image"]) as image:
            width,height = image.size
            if width*height > 80000000: raise ValueError("Image exceeds the explicit 80-million-source-pixel decode budget")
            model_h,model_w = compatible_size(height,width,max_model_pixels)
        frame = cv2.imread(sample["image"],cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != (height,width): raise ValueError(f"Cannot decode image consistently: {sample['id']}")
        prepared = frame if (model_h,model_w)==(height,width) else cv2.resize(frame,(model_w,model_h),interpolation=cv2.INTER_LINEAR)
        prediction = predictor.predict(prepared)
        raw_count = float(prediction.density.sum(dtype=np.float64))
        if not np.isfinite(prediction.density).all() or not math.isfinite(prediction.count) or not math.isclose(raw_count,prediction.count,abs_tol=1e-9,rel_tol=1e-12):
            raise ValueError("Invalid raw-density count interface; benchmark stopped")
        rows.append({**sample,"predicted_count":raw_count,"signed_error":raw_count-sample["true_count"],
                     "absolute_error":abs(raw_count-sample["true_count"]),"count_band":count_band(sample["true_count"]),
                     "source_hw":[height,width],"input_shape":list(prediction.input_shape),
                     "output_shape":list(prediction.output_shape),"inference_ms":prediction.inference_ms,
                     "image_sha256":file_sha256(Path(sample["image"]))})
        if len(rows)==1 or len(rows)%25==0: print(f"Evaluated {len(rows)}/{len(samples)} annotated images",flush=True)
    return {"status":"complete","evaluated_at_utc":datetime.now(timezone.utc).isoformat(),
            "dataset":data["dataset"],"split":data["split"],"ground_truth_source":data["ground_truth_source"],
            "ground_truth_kind":data["ground_truth_kind"],"annotations_complete":True,
            "manifest_sha256":file_sha256(Path(manifest_path)),"model":predictor.metadata,
            "frame_policy":{"max_model_pixels":max_model_pixels,"resize":"whole-frame bilinear, stride eight, no crop or count correction"},
            "metrics":count_metrics(rows),
            "by_count_band":{k:count_metrics([r for r in rows if r["count_band"]==k]) for k in ("empty_0","below_50","50_to_199","200_or_more")},
            "by_condition":{k:count_metrics([r for r in rows if r["condition"]==k]) for k in sorted({r["condition"] for r in rows})},
            "worst_samples":sorted(rows,key=lambda r:r["absolute_error"],reverse=True)[:10],
            "interpretation":"Image counts only, not ROI localization, tracking or safety validation. Bins are analysis strata, not safety thresholds.",
            "observations":rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,default=Path("outputs/checkpoints/csrnet_drive_best.pt"))
    parser.add_argument("--device",choices=["auto","cuda","cpu"],default="auto")
    parser.add_argument("--max-model-pixels",type=int,default=921600)
    parser.add_argument("--out",type=Path,required=True,help="New directory; never overwrite an evaluation")
    args = parser.parse_args()
    load_manifest(args.manifest)
    if args.out.exists(): raise FileExistsError("Choose a new --out directory; existing results are immutable")
    predictor = CSRNetPredictor(args.checkpoint,device=args.device)
    predictor.metadata["checkpoint_sha256"] = file_sha256(args.checkpoint)
    result = evaluate_manifest(args.manifest,predictor,args.max_model_pixels)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    with (args.out/"observations.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result["observations"][0]));writer.writeheader();writer.writerows(result["observations"])
    print(json.dumps({"metrics":result["metrics"],"by_count_band":result["by_count_band"],"results":str(args.out)},indent=2))


if __name__=="__main__": main()
