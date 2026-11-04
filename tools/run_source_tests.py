"""Run offline source tests with accidental Mall-loader imports prohibited."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Make source-only reproducibility enforceable even on a developer's laptop
# where the real dataset is present. Test training uses explicitly synthetic data.
sys.modules["mall_dataloader"]=None

if __name__=="__main__":
    suite=unittest.defaultTestLoader.discover(str(ROOT),pattern="test_*.py")
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
