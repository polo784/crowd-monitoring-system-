"""Real GPU sanity check on three existing Mall validation images, not a new score.

Does not train, change targets or evaluate a new independent dataset. Generated
diagnostic evidence stays under outputs, excluded from Git.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import cv2
import numpy as np
from scipy.io import loadmat

from count_benchmark import evaluate_manifest
from crowd_inference import CSRNetPredictor
from crowd_zones import zone_counts
from dashboard_service import load_config
from video_inference import file_sha256


def main():
    config=load_config()
    checkpoint=ROOT/config["checkpoint"]
    before=file_sha256(checkpoint)
    assert before==config["checkpoint_sha256"]
    base=ROOT/"cv_project/mall_dataset"
    annotations=loadmat(base/"mall_gt.mat")["frame"][0]
    samples=[]
    for number in (1401,1550,1700):
        samples.append({"id":f"mall_{number}","image":str(base/f"frames/frames/seq_{number:06d}.jpg"),
                        "true_count":len(annotations[number-1]["loc"][0][0]),"condition":"same_camera_validation"})
    out=ROOT/"outputs/benchmarks/zones_sanity_20261104"
    out.mkdir(parents=True,exist_ok=False)
    manifest=out/"manifest.json"
    manifest.write_text(json.dumps({"schema_version":1,"dataset":"Mall three-frame validation SANITY ONLY",
        "split":"validation_subset_not_independent","ground_truth_source":"Original unchanged Mall mall_gt.mat point annotations",
        "ground_truth_kind":"provided-dataset-annotations","annotations_complete":True,"samples":samples},indent=2),encoding="utf-8")
    predictor=CSRNetPredictor(checkpoint,device="auto")
    predictor.metadata["checkpoint_sha256"]=before
    result=evaluate_manifest(manifest,predictor,config["max_model_pixels"])
    checks=[]
    zones=[{"id":"all","name":"Full frame","rect":[0,0,1,1]},
           {"id":"left","name":"Left","rect":[0,0,.37,1]},
           {"id":"right","name":"Right","rect":[.37,0,1,1]}]
    for sample in samples:
        prediction=predictor.predict(cv2.imread(sample["image"]))
        regional=zone_counts(prediction.density,zones)
        difference=abs(regional["left"]+regional["right"]-regional["all"])
        assert difference<1e-9
        assert abs(regional["all"]-prediction.count)<1e-9
        checks.append({"id":sample["id"],"raw_count":prediction.count,"zones":regional,"partition_error":difference,
                       "input_shape":prediction.input_shape,"output_shape":prediction.output_shape})
    result["zone_partition_checks"]=checks
    (out/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    assert file_sha256(checkpoint)==before
    print(json.dumps({"device":predictor.metadata["device_name"],"sanity_only":True,"metrics":result["metrics"],
                      "zone_partition_checks":checks,"report":str(out/"results.json")},indent=2))


if __name__=="__main__":main()
