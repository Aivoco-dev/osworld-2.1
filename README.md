# OSWorld 2.1 Offline

**82 tasks · OpenCode 1.19.5 · Vision Large · optional continual learning**

## Start

1. Install Python 3.9+, Git, and uv. Have Node.js with npm available if OpenCode 1.19.5 is absent. The launcher installs its pinned OpenCode locally when needed.
2. Set `VISPARK_LAB_API` in this repository’s `.env`. Copy `.env.example` if needed. Accept both Hugging Face access agreements and set `HF_TOKEN` in `.env` or sign in with `uvx --from huggingface_hub hf auth login`. The launcher downloads the pinned task files and assets on first run.
3. Use the official Docker desktop image on a Linux host with KVM, or configure the official AWS desktop profile and set `OSWORLD_PROVIDER=aws` in `.env`. Use `AWS_PROFILE` to select a named AWS profile. If your environment uses the OSWorld mock websites, set `WEBSITE_HOST_SUFFIX` in the same file.
4. Double-click `Start.command` or run `./start` to run all 82 tasks. Set `BENCH_WORKERS` in `.env` to change default concurrency.

```sh
./start --plan          # List the offline tasks without API calls
./start --prepare       # Fetch the upstream version, tasks, and assets without inference
./start --task 001      # Run a single desktop task
```

OpenCode sees screenshots and selects actions. OSWorld executes the actions and runs the official evaluator. Each task starts with a fresh OpenCode session. With continual learning enabled, Vision starts each task with empty learning weights.

## Continual learning

Set `CONTINUAL_LEARNING=on` or `CONTINUAL_LEARNING=off` in this repo’s `.env` for one-click runs. The default is on. You can override it for a single run:

```sh
./start --continual-learning
./start --no-continual-learning
./start --plan --no-continual-learning
```

Both modes use the same tasks, harness, and conversation history within each task. Turning continual learning off skips the additional learning call and the use of learning weights. `run.json` records the selected mode.

## Results

Open `results/<run>/summary.json` for partial credit, strict completion, and ungraded tasks. `run.json` records the release, settings, and task IDs. The same directory contains the official task results and Vision usage.

[Official OSWorld setup](https://github.com/xlang-ai/OSWorld-V2#download-task-classes-and-assets)
