"""CPU-only CI install from this repository's locked versions; never a launcher.

Run explicitly only in a disposable CI environment. Does not modify the laptop
environment unless someone deliberately invokes this script there.
"""
from pathlib import Path
import subprocess
import sys
import tomllib


def requirements(environment=None):
    # GitHub's setup-python includes pip; the local uv environment need not.
    # Use pip's own parser without adding a runtime dependency to the project.
    from pip._vendor.packaging.markers import Marker
    from pip._vendor.packaging.requirements import Requirement
    root=Path(__file__).resolve().parents[1]
    project=tomllib.loads((root/"pyproject.toml").read_text(encoding="utf-8"))
    lock=tomllib.loads((root/"uv.lock").read_text(encoding="utf-8"))
    # Honor uv's Python/platform forks, rather than picking the last package
    # version (which might require Python 3.12 or Windows CUDA).
    versions={p["name"]:p["version"] for p in lock["package"] if "version" in p
              and (not p.get("resolution-markers") or any(Marker(m).evaluate(environment) for m in p["resolution-markers"]))}
    names=[]
    for requirement in project["project"]["dependencies"]+project["dependency-groups"]["dev"]:
        name=Requirement(requirement).name
        if name not in versions: raise ValueError(f"Missing locked CI dependency: {name}")
        names.append(name)
    cpu=[f"{name}=={versions[name].split('+')[0]}" for name in names if name in {"torch","torchvision"}]
    other=[f"{name}=={versions[name]}" for name in names if name not in {"torch","torchvision"}]
    return cpu,other


if __name__=="__main__":
    if not __import__("os").environ.get("GITHUB_ACTIONS"): raise SystemExit("This installer is for GitHub Actions only; use uv sync --frozen locally")
    cpu,other=requirements()
    subprocess.run([sys.executable,"-m","pip","install","--index-url","https://download.pytorch.org/whl/cpu",*cpu],check=True)
    subprocess.run([sys.executable,"-m","pip","install",*other],check=True)
