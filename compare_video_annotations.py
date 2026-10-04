"""Render sampled head annotations and compare with saved per-frame predictions.

Draft annotations yield provisional differences, not validated MAE/RMSE.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


REVIEW_HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Video head annotation review</title>
<style>
body{font:16px system-ui;margin:20px;background:#16212c;color:#edf4fa}
button,select,input{font:inherit}button,select{padding:6px;margin-right:8px}
.tools{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:14px 0}
#viewport{overflow:auto;max-height:70vh;border:1px solid #7b8b99;background:#000}
canvas{display:block;cursor:crosshair}#note{color:#ffbd66}p{max-width:1000px}
</style>
<h1>Review sampled frames</h1>
<p>These points are an AI visual draft, not ground truth. Some distant heads
beyond the glass may be missing. Check those regions and remove false or duplicate
marks before treating the comparison as model accuracy.</p>
<p><b>Click</b> to add a head centre. <b>Right-click</b> a point to remove it.
<b>Shift-click</b> an existing point to switch orange (uncertain) / cyan (candidate).
Use the zoom control and horizontal scrollbar to inspect the full image.</p>
<div class="tools">
<select id="frame"></select><select id="zoom"><option value="0.5">50%</option>
<option value="0.75">75%</option><option value="1">100%</option>
<option value="1.5">150%</option></select>
<label><input id="ids" type="checkbox">Show point IDs</label>
<label><input id="uncertain" type="checkbox">New point is uncertain</label>
<button id="save">Export annotations JSON</button>
<label>Load edited JSON <input id="load" type="file" accept=".json"></label>
</div>
<label><input id="reviewed" type="checkbox">I reviewed the entire frame, resolved
all orange points, and checked the distant crowd. These annotations are complete.</label>
<p id="counts"></p><p id="note"></p><div id="viewport"><canvas id="canvas"></canvas></div>
<p>Export the corrected JSON and rerun the comparison command shown in README.txt.
Only frames marked complete and reviewed, with no unresolved candidates, qualify
for MAE/RMSE. A point set must be checked against the image; its size alone does
not prove that it is correct.</p>
<script>
let state=__ANNOTATIONS__;
const predictions=__PREDICTIONS__;
const canvas=document.querySelector('#canvas'), ctx=canvas.getContext('2d');
const selector=document.querySelector('#frame'), reviewed=document.querySelector('#reviewed');
const image=new Image(); let current=0;
function frame(){return state.frames[current];}
function build(){selector.replaceChildren(); state.frames.forEach((f,i)=>{
 let o=document.createElement('option');o.value=i;o.textContent=`Frame ${f.frame_index} (${predictions[f.frame_index].timestamp_seconds}s)`;selector.append(o);
});current=0;selector.value='0';openFrame();}
function openFrame(){image.src=frame().image;reviewed.checked=!!(frame().reviewed&&frame().complete);}
function draw(){if(!image.complete||!image.naturalWidth)return;
 const f=frame();canvas.width=f.width;canvas.height=f.height;
 const z=Number(document.querySelector('#zoom').value);canvas.style.width=`${f.width*z}px`;
 canvas.style.height=`${f.height*z}px`;ctx.drawImage(image,0,0,f.width,f.height);
 for(const region of f.unresolved_regions||[]){ctx.beginPath();region.polygon.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.strokeStyle='#ffc063';ctx.lineWidth=3;ctx.setLineDash([12,8]);ctx.stroke();ctx.setLineDash([]);}
 for(const p of f.points){ctx.beginPath();ctx.arc(p.x,p.y,7,0,Math.PI*2);ctx.strokeStyle=p.uncertain?'#ffae36':'#00edff';ctx.lineWidth=2;ctx.stroke();if(document.querySelector('#ids').checked){ctx.font='13px sans-serif';ctx.fillStyle=ctx.strokeStyle;ctx.fillText(p.id,p.x+9,p.y-5);}}
 const n=f.points.length,u=f.points.filter(p=>p.uncertain).length,pred=Number(predictions[f.frame_index].predicted_count);
 document.querySelector('#counts').textContent=`Draft candidates: ${n} (${u} uncertain) | CSRNet: ${pred.toFixed(2)} | Model minus draft: ${(pred-n).toFixed(2)}`;
 document.querySelector('#note').textContent=f.reviewed&&f.complete?'Human review recorded for this frame.':'Unreviewed / incomplete draft. Differences are not validated counting errors.';
 reviewed.checked=!!(f.reviewed&&f.complete);
}
function invalidate(){frame().reviewed=false;frame().complete=false;}
function position(e){const r=canvas.getBoundingClientRect();return [(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height];}
function nearest(x,y){let best=-1,d=Infinity;frame().points.forEach((p,i)=>{let n=Math.hypot(p.x-x,p.y-y);if(n<d){d=n;best=i;}});return d<20?best:-1;}
canvas.onclick=e=>{let[x,y]=position(e),i=nearest(x,y);if(e.shiftKey){if(i>=0)frame().points[i].uncertain=!frame().points[i].uncertain;}else{let id=Math.max(0,...frame().points.map(p=>p.id))+1;frame().points.push({id,x:Math.round(x),y:Math.round(y),uncertain:document.querySelector('#uncertain').checked});}invalidate();draw();};
canvas.oncontextmenu=e=>{e.preventDefault();let[x,y]=position(e),i=nearest(x,y);if(i>=0){frame().points.splice(i,1);invalidate();draw();}};
reviewed.onchange=()=>{if(reviewed.checked&&frame().points.some(p=>p.uncertain)){reviewed.checked=false;document.querySelector('#note').textContent='Resolve the orange candidates first: shift-click to retain as a candidate, or right-click to remove.';return;}frame().reviewed=reviewed.checked;frame().complete=reviewed.checked;if(reviewed.checked)frame().unresolved_regions=[];draw();};
image.onload=draw;selector.onchange=()=>{current=Number(selector.value);openFrame();};
document.querySelector('#zoom').onchange=draw;document.querySelector('#ids').onchange=draw;
document.querySelector('#save').onclick=()=>{state.status=state.frames.every(f=>f.reviewed&&f.complete&&!f.points.some(p=>p.uncertain))?'human_reviewed':'draft_not_ground_truth';const blob=new Blob([JSON.stringify(state,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='annotations_reviewed.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
document.querySelector('#load').onchange=async e=>{try{const candidate=JSON.parse(await e.target.files[0].text());if(!Array.isArray(candidate.frames)||candidate.frames.length!==Object.keys(predictions).length||new Set(candidate.frames.map(f=>f.frame_index)).size!==candidate.frames.length||candidate.frames.some(f=>!predictions[f.frame_index]||!Array.isArray(f.points)))throw Error('Expected the original sampled frames');state=candidate;build();}catch(error){document.querySelector('#note').textContent=error.message;}};
build();
</script></html>'''


def compare(annotations_path: Path, measurements_path: Path, output_dir: Path) -> dict:
    annotations_path = annotations_path.resolve()
    document = json.loads(annotations_path.read_text(encoding="utf-8"))
    with measurements_path.open(newline="", encoding="utf-8") as stream:
        prediction_rows = list(csv.DictReader(stream))
    predictions = {int(row["frame_index"]): row for row in prediction_rows}
    if len(predictions) != len(prediction_rows):
        raise ValueError("Predictions contain duplicate frame indexes")
    frames = document["frames"]
    if len({f["frame_index"] for f in frames}) != len(frames) or not frames:
        raise ValueError("Annotation frames must be non-empty and unique")
    results = []
    overlays = []
    for frame in frames:
        index = frame["frame_index"]
        if index not in predictions:
            raise ValueError(f"No saved model prediction for frame {index}")
        image_path = (annotations_path.parent / frame["image"]).resolve()
        if not image_path.is_relative_to(annotations_path.parent):
            raise ValueError("Annotation images must be inside the annotation directory")
        image = cv2.imread(str(image_path))
        if image is None or image.shape[:2] != (frame["height"], frame["width"]):
            raise ValueError(f"Annotation image/dimensions invalid: {image_path}")
        points = frame["points"]
        if len({p["id"] for p in points}) != len(points):
            raise ValueError("Point IDs must be unique within each frame")
        for point in points:
            x, y = point["x"], point["y"]
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < frame["width"] and 0 <= y < frame["height"]):
                raise ValueError(f"Invalid head point in frame {index}")
            cv2.circle(image, (round(x), round(y)), 8,
                       (0, 165, 255) if point.get("uncertain", False) else (255, 255, 0), 2)
        uncertain = sum(bool(p.get("uncertain", False)) for p in points)
        qualified = bool(frame.get("reviewed") and frame.get("complete") and
                         not frame.get("unresolved_regions") and uncertain == 0)
        prediction = float(predictions[index]["predicted_count"])
        if not math.isfinite(prediction):
            raise ValueError("Model count is non-finite")
        count = len(points)
        result = {
            "frame_index": index,
            "timestamp_seconds": float(predictions[index]["timestamp_seconds"]),
            "annotation_candidates": count,
            "uncertain_candidates": uncertain,
            "predicted_count": prediction,
            "model_minus_annotations": prediction - count,
            "absolute_difference": abs(prediction - count),
            "qualified_for_accuracy_metrics": qualified,
        }
        results.append(result)
        label = "REVIEWED" if qualified else "AI DRAFT - NOT GROUND TRUTH"
        cv2.rectangle(image, (0, 0), (min(frame["width"], 1300), 92), (0, 0, 0), -1)
        cv2.putText(image, f"{label} | Frame {index} | Time {result['timestamp_seconds']:.2f}s",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"Head candidates: {count} ({uncertain} uncertain) | CSRNet: {prediction:.2f}",
                    (12, 66), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 1, cv2.LINE_AA)
        overlays.append((index, image))
    all_reviewed = all(row["qualified_for_accuracy_metrics"] for row in results)
    mean_difference = sum(row["absolute_difference"] for row in results) / len(results)
    report = {
        "status": "reviewed_comparison" if all_reviewed else "provisional_comparison",
        "annotations": str(annotations_path), "measurements": str(measurements_path.resolve()),
        "sampled_frames": results,
        "mae": mean_difference if all_reviewed else None,
        "rmse": math.sqrt(sum(row["model_minus_annotations"]**2 for row in results) / len(results)) if all_reviewed else None,
        "mean_absolute_difference_from_draft": None if all_reviewed else mean_difference,
        "limitations": [
            "AI points may include misses, duplicates and incorrect locations; human review is required.",
            "Distant crowd through glass is incompletely annotated in the initial draft.",
            "Model minus draft is not evidence of model overcounting while annotations are incomplete.",
            "Three frames from one 11-second clip do not establish general model accuracy.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for index, image in overlays:
        path = output_dir / f"frame_{index:06d}_annotated.jpg"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not save overlay: {path}")
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    # The editor lives beside the original images, with no server or network access.
    html = REVIEW_HTML.replace("__ANNOTATIONS__", json.dumps(document).replace("<", "\\u003c"))
    html = html.replace("__PREDICTIONS__", json.dumps({row["frame_index"]: predictions[row["frame_index"]] for row in results}).replace("<", "\\u003c"))
    (annotations_path.parent / "review.html").write_text(html, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.annotations, args.measurements, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
