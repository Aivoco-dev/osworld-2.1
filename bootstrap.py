"""Checkout the published benchmark revision and keep results local."""

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent


def checkout(url, revision):
    upstream = REPO / ".upstream"
    if not upstream.exists():
        upstream.mkdir()
        try:
            subprocess.run(["git", "init", "-q", str(upstream)], check=True)
        except BaseException:
            upstream.rmdir()
            raise
    git = ["git", "-C", str(upstream)]
    current = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if current != revision:
        subprocess.run([*git, "fetch", "--depth=1", url, revision], check=True)
        subprocess.run([*git, "checkout", "--detach", revision], check=True)
    return upstream


def require_tools(*names):
    missing = [name for name in names if not shutil.which(name)]
    if missing:
        raise RuntimeError("Install the required host tools: " + ", ".join(missing))


def ensure_env():
    path = REPO / ".env"
    if not path.exists():
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        with os.fdopen(fd, "w") as file:
            file.write((REPO / ".env.example").read_text())


def result_directory(*, benchmark, harness, release, tasks, workers, **options):
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    path = REPO / "results" / stamp
    path.mkdir(parents=True)
    metadata = {
        "benchmark": benchmark,
        "harness": harness,
        "release": release,
        "model": "vispark/vision-large",
        "continual_learning": True,
        "started_utc": now.isoformat(),
        "task_count": len(tasks),
        "tasks": tasks,
        "workers": workers,
        **options,
    }
    (path / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{benchmark} | {len(tasks)} tasks | {harness} | {workers} worker(s)", flush=True)
    print(f"Results: {path}", flush=True)
    return path
