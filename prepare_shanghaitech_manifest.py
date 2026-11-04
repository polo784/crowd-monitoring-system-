"""Read existing ShanghaiTech points into a count benchmark. No download/training."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def create_manifest(root,part,split,output):
    root,output = Path(root).resolve(),Path(output).resolve()
    if part not in {"A","B"} or split not in {"train","test"}: raise ValueError("Choose part A/B and split train/test")
    if output.exists(): raise FileExistsError("Manifest already exists; choose a new path")
    folder = root/f"part_{part}_final"/f"{split}_data"
    images = sorted((folder/"images").glob("IMG_*.jpg"))
    if not images: raise FileNotFoundError(f"No original-layout images found in {folder/'images'}")
    samples=[]
    for image in images:
        annotation=folder/"ground-truth"/f"GT_{image.stem}.mat"
        info=loadmat(annotation,simplify_cells=True).get("image_info")
        if not isinstance(info,dict) or "location" not in info: raise ValueError(f"Unexpected annotation schema: {annotation}")
        points=np.asarray(info["location"],dtype=np.float64)
        if points.size==0: points=points.reshape(0,2)
        elif points.ndim==1 and points.size==2: points=points.reshape(1,2)
        if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all(): raise ValueError(f"Invalid annotations: {annotation}")
        count=len(points)
        if "number" in info and (np.asarray(info["number"]).size!=1 or float(np.asarray(info["number"]).item())!=count):
            raise ValueError(f"Point count disagrees with number: {annotation}")
        samples.append({"id":f"SHA{part}_{split}_{image.stem}","image":str(image),"true_count":count,
                        "condition":"unspecified","annotation":str(annotation)})
    result={"schema_version":1,"dataset":f"ShanghaiTech Part {part}","split":split,
            "annotations_complete":True,"ground_truth_kind":"provided-dataset-annotations",
            "ground_truth_source":"https://github.com/desenzhou/ShanghaiTechDataset; local original point annotations",
            "samples":samples}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--part",choices=["A","B"],default="A")
    parser.add_argument("--split",choices=["train","test"],default="test")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=create_manifest(args.root,args.part,args.split,args.out)
    print(f"Read {len(result['samples'])} unchanged annotated images; manifest: {args.out}")


if __name__=="__main__": main()
