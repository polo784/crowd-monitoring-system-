"""Download the official HT21-01 raw video and labels without touching training data."""

import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parent / "videos" / "ht21"
FILES = {
    "HT21-01-raw.mp4": "https://motchallenge.net/sequenceVideos/HT21-01-raw.mp4",
    "HT21Labels.zip": "https://motchallenge.net/data/HT21Labels.zip",
}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in FILES.items():
        path = ROOT / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        partial = path.with_suffix(path.suffix + ".partial")
        started = time.perf_counter()
        request = urllib.request.Request(url, headers={"User-Agent": "Mall-CSRNet-research/1.0"})
        print(f"Downloading {url}", flush=True)
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as output:
            expected = int(response.headers.get("Content-Length", 0))
            size, digest = 0, hashlib.sha256()
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
                size += len(block)
                print(f"  {size / 1048576:.1f} MiB / {expected / 1048576:.1f} MiB", flush=True)
        if expected and size != expected:
            raise RuntimeError(f"Incomplete download: {name}")
        partial.rename(path)
        records.append({"file": name, "url": url, "bytes": size,
                        "sha256": digest.hexdigest(), "download_seconds": time.perf_counter() - started})
    with zipfile.ZipFile(ROOT / "HT21Labels.zip") as archive:
        print("Annotation archive members:", flush=True)
        for member in archive.infolist():
            print(member.filename, member.file_size, flush=True)
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity failure: {bad}")
    (ROOT / "download_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
