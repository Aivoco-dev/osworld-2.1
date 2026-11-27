import argparse
import json
import math
import os
import subprocess
import sys

from bootstrap import REPO, ensure_env, require_tools, checkout, result_directory
from cl_adapter import load_key, read_env, add_continual_learning_argument, resolve_continual_learning
from opencode_driver import OPENCODE_VERSION, prepare_opencode

URL = "https://github.com/xlang-ai/OSWorld-V2"
REVISION = "3d778a3c9a34a079316f70df023b166700445792"
RELEASE = "osworld-v2.1"


def main():
    settings = read_env(REPO)
    parser = argparse.ArgumentParser(description="OSWorld 2.1 offline with OpenCode and Vision")
    parser.add_argument("--workers", type=int, default=settings.get("BENCH_WORKERS") or "1")
    parser.add_argument("--provider", choices=["docker", "aws"], default=settings.get("OSWORLD_PROVIDER") or "docker")
    parser.add_argument("--task", action="append")
    parser.add_argument("--prepare", action="store_true", help="Download versioned gated tasks and assets")
    parser.add_argument("--plan", action="store_true", help="Show the selected task list without inference")
    add_continual_learning_argument(parser)
    args = parser.parse_args()
    try:
        continual_learning = resolve_continual_learning(args.continual_learning, settings)
    except ValueError as error:
        parser.error(str(error))
    tasks = json.loads((REPO / "tasks.json").read_text())["tasks"]
    if len(set(tasks)) != 82:
        raise RuntimeError("Expected the 82-task OSWorld offline manifest")
    selected = args.task or tasks
    if args.workers < 1 or len(set(selected)) != len(selected) or not set(selected) <= set(tasks):
        parser.error("Use positive workers and IDs from tasks.json")
    if args.provider not in ("docker", "aws"):
        parser.error("OSWORLD_PROVIDER must be docker or aws")
    if args.plan:
        print(json.dumps({
            "release": RELEASE, "harness": "OpenCode + Vision", "split": "offline",
            "provider": args.provider, "workers": args.workers, "continual_learning": continual_learning, "task_count": len(selected),
            "tasks": selected,
        }, indent=2))
        return 0
    require_tools("git", "uv")
    if not args.prepare:
        ensure_env()
        load_key(REPO)
        if args.provider == "docker":
            require_tools("docker")
            if subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
                raise RuntimeError("Start the Docker engine and check that docker info succeeds")
    source = checkout(URL, REVISION)
    env = os.environ | {"PYTHONPATH": os.pathsep.join([str(REPO), str(source)]), "OSWORLD_BENCHMARK_RELEASE": RELEASE}
    for name in ("HF_TOKEN", "AWS_PROFILE", "WEBSITE_HOST_SUFFIX"):
        if settings.get(name):
            env[name] = settings[name]
    assets = source / "cache" / "osworld-v2.1-assets"

    def uv(*command):
        subprocess.run(["uv", *command], cwd=source, env=env, check=True)

    uv("sync", "--frozen")
    ready = assets / ".vispark-ready"
    if args.prepare or not ready.is_file():
        uv("run", "--no-sync", "python", "scripts/tools/download_osworld_v2_tasks.py", "--benchmark-release", RELEASE)
        uv("run", "--no-sync", "python", "scripts/tools/download_osworld_v2_assets.py",
           "--benchmark-release", RELEASE, "--target-dir", str(assets))
        ready.touch()
    if args.prepare:
        return 0
    env["OSWORLD_FILE_BASE_URL"] = str(assets)
    env["VISION_OPENCODE_BIN"] = str(prepare_opencode(REPO))
    output = result_directory(
        benchmark="OSWorld", harness="OpenCode + Vision",
        release=RELEASE, tasks=selected, workers=args.workers, continual_learning=continual_learning, split="offline",
        revision=REVISION, provider=args.provider, opencode_version=OPENCODE_VERSION, max_steps=500,
    )
    env["VISION_CONTINUAL_LEARNING"] = "on" if continual_learning else "off"
    env["VISION_OSWORLD_USAGE_DIR"] = str(output / "vision-usage")
    task_map = output / "tasks.json"
    task_map.write_text(json.dumps({"offline": selected}, indent=2))
    uv("run", "--no-sync", "python", str(REPO / "osworld_runner.py"),
       "--model", "vision-large-cl", "--provider_name", args.provider, "--action_space", "pyautogui",
       "--headless", "--max_steps", "500", "--num_envs", str(args.workers),
       "--test_all_meta_path", str(task_map), "--result_dir", str(output))
    scores = {}
    result_root = output / "pyautogui" / "screenshot" / "vision-large-cl" / "offline"
    for task in selected:
        try:
            score = float((result_root / task / "result.txt").read_text())
        except (OSError, ValueError):
            continue
        if math.isfinite(score):
            scores[task] = score
    summary = {
        "planned": len(selected), "graded": len(scores),
        "ungraded": [task for task in selected if task not in scores],
        "mean_partial_credit": sum(scores.values()) / len(selected),
        "strict_pass_rate": sum(score == 1.0 for score in scores.values()) / len(selected),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(len(scores) != len(selected))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Benchmark startup or run failed: {error}", file=sys.stderr)
        raise SystemExit(1)
