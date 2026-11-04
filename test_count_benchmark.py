"""Offline annotated fixtures only; these are not real-dataset accuracy results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.io import savemat

from count_benchmark import count_band,count_metrics,evaluate_manifest,load_manifest
from prepare_shanghaitech_manifest import create_manifest
from test_dashboard import FakePredictor


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.scratch=tempfile.TemporaryDirectory();self.root=Path(self.scratch.name)
        self.manifest=self.root/"counts.json"
        self.data={"schema_version":1,"dataset":"explicit synthetic fixture","split":"diagnostic",
                   "ground_truth_source":"synthetic unit test, not real annotations","annotations_complete":True,
                   "ground_truth_kind":"provided-dataset-annotations","samples":[]}
        for i,count in enumerate([0,2,50,200]):
            image=self.root/f"{i}.png";Image.new("RGB",(48,32),(i,1,1)).save(image)
            self.data["samples"].append({"id":str(i),"image":image.name,"true_count":count,"condition":"night" if i%2 else "day"})
        self.save()
    def save(self): self.manifest.write_text(json.dumps(self.data))
    def tearDown(self):self.scratch.cleanup()
    def test_metrics_and_strata_are_count_errors(self):
        result=evaluate_manifest(self.manifest,FakePredictor())
        self.assertEqual(result["status"],"complete")
        self.assertEqual(result["metrics"]["samples"],4)
        self.assertEqual(result["metrics"]["mae"],62.) # errors +4,+2,-46,-196
        self.assertEqual(result["metrics"]["signed_bias"],-59.)
        self.assertAlmostEqual(result["metrics"]["rmse"],float(np.sqrt((16+4+2116+38416)/4)))
        self.assertEqual([result["by_count_band"][k]["samples"] for k in ("empty_0","below_50","50_to_199","200_or_more")],[1]*4)
        self.assertEqual(result["worst_samples"][0]["id"],"3")
        self.assertEqual(len(result["observations"][0]["image_sha256"]),64)
    def test_incomplete_duplicate_missing_and_invalid_ground_truth_rejected(self):
        original=copy.deepcopy(self.data)
        for mutation in (lambda d:d.update(annotations_complete=False),
                         lambda d:d["samples"].append(copy.deepcopy(d["samples"][0])),
                         lambda d:d["samples"][0].update(true_count=True),
                         lambda d:d["samples"][0].update(true_count=float("nan")),
                         lambda d:d["samples"][0].update(image="missing.png"),
                         lambda d:d.update(ground_truth_kind="AI-draft")):
            self.data=copy.deepcopy(original);mutation(self.data);self.save()
            with self.assertRaises((ValueError,FileNotFoundError)):load_manifest(self.manifest)
    def test_raw_sum_interface_and_pixel_budget_rejected(self):
        predictor=FakePredictor();predict=predictor.predict
        def wrong(frame):
            result=predict(frame);result.count=7.;return result
        predictor.predict=wrong
        with self.assertRaisesRegex(ValueError,"raw-density"):evaluate_manifest(self.manifest,predictor)
        with self.assertRaises(ValueError):evaluate_manifest(self.manifest,FakePredictor(),True)
        self.assertIsNone(count_metrics([]))
    def test_shanghaitech_original_schema_zero_single_and_multiple_points(self):
        base=self.root/"part_A_final/test_data";(base/"images").mkdir(parents=True);(base/"ground-truth").mkdir()
        for i,number in enumerate([0,1,3]):
            Image.new("RGB",(48,32)).save(base/f"images/IMG_{i}.jpg")
            # Same nested MATLAB cell/struct shape as the original dataset.
            info=np.empty((1,1),dtype=object)
            info[0,0]={"location":np.ones((number,2)),"number":number}
            savemat(base/f"ground-truth/GT_IMG_{i}.mat",{"image_info":info})
        output=self.root/"shanghai.json";result=create_manifest(self.root,"A","test",output)
        self.assertEqual([s["true_count"] for s in result["samples"]],[0,1,3])
        self.assertEqual(load_manifest(output)[0]["split"],"test")
        with self.assertRaises(FileExistsError):create_manifest(self.root,"A","test",output)


if __name__=="__main__":unittest.main()
