"""Run the official desktop evaluator with a Vision-backed OpenCode agent."""

import atexit
import json
import logging
import os
import signal
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from bootstrap import REPO
from cl_adapter import bridge, load_key, parse_continual_learning
from opencode_driver import OpenCode

sys.path.insert(0, str(Path.cwd() / "scripts" / "python"))
import run_multienv as upstream

ACTIONS = (
    "Return JSON only with this shape: "
    '{"text":"","actions":[{"name":"click","arguments":{"x":500,"y":500}}]}. '
    "Coordinates are integers from 0 to 1000. You may batch related actions. "
    "Use these actions: click, double_click, right_click, move (x, y); "
    "drag (x, y, end_x, end_y); scroll (x, y, delta); type (text); "
    "press (key); hotkey (keys, a list of key names); wait, done, infeasible (no arguments). "
    "Use text with an empty action list if you need to ask the simulated user a question. "
    "Use done once the task is complete."
)
POINT_ACTIONS = {"click": "click", "double_click": "doubleClick", "right_click": "rightClick", "move": "moveTo"}


def desktop_action(name, arguments, width, height):
    if not isinstance(arguments, dict):
        raise ValueError("Action arguments must be an object")

    def point(x="x", y="y"):
        def pixel(value, size):
            if type(value) is not int or not 0 <= value <= 1000:
                raise ValueError("Coordinates must be integers from 0 to 1000")
            return min(size - 1, value * size // 1000)
        return pixel(arguments[x], width), pixel(arguments[y], height)

    if name in POINT_ACTIONS:
        x, y = point()
        return f"pyautogui.{POINT_ACTIONS[name]}({x}, {y})"
    if name == "drag":
        x, y = point()
        end_x, end_y = point("end_x", "end_y")
        return f"pyautogui.moveTo({x}, {y}); pyautogui.dragTo({end_x}, {end_y}, duration=0.5)"
    if name == "scroll":
        x, y = point()
        delta = arguments["delta"]
        if type(delta) is not int or not -100 <= delta <= 100:
            raise ValueError("Scroll delta must be an integer from -100 to 100")
        return f"pyautogui.scroll({delta}, x={x}, y={y})"
    if name in ("type", "press"):
        value = arguments["text" if name == "type" else "key"]
        if not isinstance(value, str):
            raise ValueError("Expected a string")
        method = "write" if name == "type" else "press"
        return f"pyautogui.{method}({value!r})"
    if name == "hotkey":
        keys = arguments["keys"]
        if not isinstance(keys, list) or not 1 <= len(keys) <= 5 or any(not isinstance(key, str) for key in keys):
            raise ValueError("Expected one to five keys")
        return "pyautogui.hotkey(" + ", ".join(repr(key) for key in keys) + ")"
    if name in ("wait", "done", "infeasible"):
        return {"wait": "WAIT", "done": "DONE", "infeasible": "FAIL"}[name]
    raise ValueError("Unknown desktop action")


class OpenCodeDesktopAgent:
    def __init__(self, **kwargs):
        self._logger = logging.getLogger(__name__)
        self.close_task = None
        self.last_error = ""

    def reset(self, runtime_logger=None):
        if self.close_task:
            self.close_task()
            self.close_task = None
        self.last_error = ""
        if runtime_logger:
            self._logger = runtime_logger

    def _start_task(self):
        profile = tempfile.TemporaryDirectory(prefix="vision-osworld-")
        continual_learning = parse_continual_learning(os.environ.get("VISION_CONTINUAL_LEARNING", "on"))
        adapter = bridge(load_key(REPO), limit=500, continual_learning=continual_learning)
        url, token, vision = adapter.__enter__()
        try:
            self.cli = OpenCode(Path(profile.name), url, token)
        except BaseException:
            adapter.__exit__(None, None, None)
            profile.cleanup()
            raise
        report_dir = Path(os.environ["VISION_OSWORLD_USAGE_DIR"])
        report_dir.mkdir(parents=True, exist_ok=True)
        self.usage_path = report_dir / (uuid4().hex + ".json")
        self.vision = vision

        def close_task():
            atexit.unregister(close_task)
            adapter.__exit__(None, None, None)
            profile.cleanup()

        self.close_task = close_task
        atexit.register(close_task)

    def predict(self, instruction, obs):
        if self.close_task is None:
            self._start_task()
        if not isinstance(obs.get("screenshot"), bytes):
            raise ValueError("Desktop screenshot is missing")
        image = self.cli.directory / (uuid4().hex + ".png")
        image.write_bytes(obs["screenshot"])
        prompt = ["Inspect the current screenshot and continue the task.", ACTIONS]
        if self.cli.session is None:
            prompt.insert(0, f"Task: {instruction}\nTask date: {getattr(self, 'task_current_date', None)}")
        if obs.get("user_response"):
            prompt.append("User response: " + str(obs["user_response"]))
        if self.last_error:
            prompt.append("The last action batch was invalid. Correct the JSON and try again.")
        try:
            response = self.cli.prompt("\n".join(prompt), [image])
        finally:
            image.unlink()
            self.usage_path.write_text(json.dumps(self.vision.report(), indent=2) + "\n")
        try:
            if response.startswith("```") and response.endswith("```"):
                response = response.split("\n", 1)[1].rsplit("\n", 1)[0]
            message = json.loads(response)
            text = message.get("text", "")
            actions = message.get("actions", [])
            if not isinstance(text, str) or not isinstance(actions, list) or len(actions) > 10 or not (text or actions):
                raise ValueError("Expected a question or one to ten actions")
            parsed = [
                desktop_action(action["name"], action["arguments"], 1920, 1080)
                for action in actions
            ]
            self.last_error = ""
            return text, parsed
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            self.last_error = str(error)
            self._logger.warning("Invalid desktop action batch: %s", error)
            return "", ["WAIT"]


upstream.PromptAgent = OpenCodeDesktopAgent


def main():
    args = upstream.config()
    signal.signal(signal.SIGINT, upstream.signal_handler)
    signal.signal(signal.SIGTERM, upstream.signal_handler)
    with open(args.test_all_meta_path, encoding="utf-8") as manifest:
        tasks = json.load(manifest)
    upstream.test(args, tasks)


if __name__ == "__main__":
    main()
