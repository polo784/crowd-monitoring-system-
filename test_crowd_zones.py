"""Math and API regression tests for count-preserving rectangular regions."""
import copy
import threading
import unittest

import numpy as np
from fastapi.testclient import TestClient

from crowd_zones import validate_zones, zone_counts
from dashboard import create_app
from dashboard_service import RuntimeProblem
import test_dashboard as fixtures


def zone(identity,rect): return {"id":identity,"name":identity,"rect":rect}


class ZoneMathTests(unittest.TestCase):
    def test_full_frame_and_partition_equal_signed_raw_sum(self):
        density=np.array([[2,-3],[4,1]],dtype=np.float32)
        original=density.copy()
        counts=zone_counts(density,[zone("all",[0,0,1,1]),zone("left",[0,0,.5,1]),zone("right",[.5,0,1,1])])
        self.assertEqual(counts,{"all":4.,"left":6.,"right":-2.})
        self.assertEqual(counts["left"]+counts["right"],counts["all"])
        np.testing.assert_array_equal(density,original)

    def test_fractional_boundaries_and_overlap(self):
        density=np.array([[8.]],dtype=np.float32)
        counts=zone_counts(density,[zone("quarter",[0,0,.5,.5]),zone("same",[0,0,.5,.5])])
        self.assertEqual(counts,{"quarter":2.,"same":2.})
        density=np.arange(35,dtype=np.float32).reshape(5,7)-20
        parts=zone_counts(density,[zone("one",[0,0,.37,1]),zone("two",[.37,0,1,1])])
        self.assertAlmostEqual(sum(parts.values()),float(density.sum(dtype=np.float64)),places=10)

    def test_invalid_zones_and_density_fail(self):
        for rect in ([.5,0,.4,1],[0,0,0,1],[-.1,0,1,1],[0,0,float("nan"),1],[False,0,1,1]):
            with self.assertRaises(ValueError): validate_zones([zone("a",rect)])
        for zones in ([zone("a",[0,0,1,1])]*2,[zone("a",[0,0,1,1])]*7,{},None):
            with self.assertRaises(ValueError): validate_zones(zones)
        with self.assertRaises(ValueError): validate_zones([{"id":"bad","name":"\n","rect":[0,0,1,1]}])
        with self.assertRaises(ValueError): zone_counts(np.array([[float("nan")]]),[])


class ZoneRuntimeTests(unittest.TestCase):
    setUp=fixtures.DashboardTests.setUp
    tearDown=fixtures.DashboardTests.tearDown
    play=fixtures.DashboardTests.play
    infer=fixtures.DashboardTests.infer

    def test_zones_api_exports_and_seek_preservation(self):
        zones=[zone("left",[0,0,.5,1]),zone("right",[.5,0,1,1])]
        with TestClient(create_app(self.config,self.runtime)) as client:
            command={"session_id":self.session["id"],"generation":1,"action":"zones","timestamp":0,"playing":False,"zones":zones}
            response=client.post("/api/control",json=command)
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()["zones"],zones)
            result=self.infer(.1,epoch=1,snapshot=True)
            self.assertEqual(result["count"],4.)
            self.assertEqual(result["zone_counts"],{"left":6.,"right":-2.})
            exported=client.get("/api/export").json()
            self.assertEqual(exported["metadata"]["zones"],zones)
            self.assertEqual(exported["observations"][0]["zone_counts"],result["zone_counts"])
            self.runtime.control(self.session["id"],2,"seek",1,False)
            self.assertEqual(self.runtime.session.zones,zones)
            self.assertEqual(list(self.runtime.session.history),[])


    def test_zone_edit_invalidates_inflight_and_resets_segment(self):
        self.play();self.infer()
        old_segment=self.runtime.session.segment_id
        predictor=self.runtime.predictor
        predictor.entered,predictor.release=threading.Event(),threading.Event()
        errors=[]
        def work():
            try:self.infer(.2)
            except RuntimeProblem as error:errors.append(error)
        worker=threading.Thread(target=work);worker.start()
        self.assertTrue(predictor.entered.wait(3))
        self.runtime.control(self.session["id"],2,"zones",.2,False,[zone("all",[0,0,1,1])])
        predictor.release.set();worker.join(5)
        self.assertFalse(worker.is_alive());self.assertEqual(errors[0].status,409)
        self.assertNotEqual(self.runtime.session.segment_id,old_segment)
        self.assertIsNone(self.runtime.session.latest)
        self.assertEqual(list(self.runtime.session.history),[])


    def test_invalid_edit_is_atomic_and_new_video_resets_zones(self):
        self.play();self.infer()
        with self.assertRaises(RuntimeProblem):self.runtime.control(self.session["id"],2,"zones",.2,False,[zone("a",[1,0,0,1])])
        self.assertEqual(self.runtime.session.generation,1)
        self.assertEqual(len(self.runtime.session.history),1)
        self.runtime.control(self.session["id"],2,"zones",.2,False,[zone("all",[0,0,1,1])])
        self.runtime.select_video(self.root/"other.mp4","other.mp4",self.metadata)
        self.assertEqual(self.runtime.session.zones,[])


if __name__=="__main__":unittest.main()
