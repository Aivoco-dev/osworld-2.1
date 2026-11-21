"""Headless OpenCode with a fresh profile per benchmark attempt."""

import json
import os
import shutil
import subprocess

from cl_adapter import MODEL

OPENCODE_VERSION = "1.19.5"


def prepare_opencode(repo):
    """Reuse the verified CLI version or install it into this repository."""
    tools = repo / ".tools"
    local_binary = tools / "node_modules" / ".bin" / "opencode"
    for candidate in (str(local_binary), shutil.which("opencode")):
        if not candidate or not os.access(candidate, os.X_OK):
            continue
        release = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=15)
        if release.returncode == 0 and release.stdout.strip() == OPENCODE_VERSION:
            return candidate
    if not shutil.which("npm"):
        raise RuntimeError(f"Install Node.js with npm, or install OpenCode {OPENCODE_VERSION} on PATH")
    subprocess.run([
        "npm", "install", "--prefix", str(tools), "--no-save", "--no-package-lock",
        "--no-audit", "--no-fund", f"opencode-ai@{OPENCODE_VERSION}",
    ], check=True)
    release = subprocess.run([str(local_binary), "--version"], capture_output=True, text=True, timeout=15)
    if release.returncode or release.stdout.strip() != OPENCODE_VERSION:
        raise RuntimeError("The local OpenCode installation did not report the pinned version")
    return str(local_binary)


class OpenCode:
    def __init__(self, directory, url, token, mcp_command=None):
        self.directory = directory
        self.session = None
        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {"vision": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": url, "apiKey": token, "timeout": 360000},
                "models": {MODEL: {
                    "name": "Vision Large CL", "attachment": True,
                    "modalities": {"input": ["text", "image"], "output": ["text"]},
                    "limit": {"context": 980000, "output": 65536},
                }},
            }},
            "enabled_providers": ["vision"],
            "permission": {"*": "deny"},
            "autoupdate": False,
            "share": "disabled",
        }
        if mcp_command:
            config["mcp"] = {"bench": {"type": "local", "command": mcp_command, "enabled": True}}
            config["permission"]["bench_*"] = "allow"
        config_path = directory / "opencode.json"
        config_path.write_text(json.dumps(config))
        config_path.chmod(0o600)
        self.env = {
            key: value for key, value in os.environ.items()
            if key in ("PATH", "HOME", "LANG", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR", "LC_ALL")
        }
        self.env.update({
            "OPENCODE_CONFIG": str(config_path),
            "OPENCODE_DISABLE_AUTOCOMPACT": "true",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "true",
            "XDG_CONFIG_HOME": str(directory / "config"),
            "XDG_DATA_HOME": str(directory / "data"),
            "XDG_STATE_HOME": str(directory / "state"),
            "XDG_CACHE_HOME": str(directory / "cache"),
        })
        self.bin = os.environ.get("VISION_OPENCODE_BIN") or shutil.which("opencode")
        if not self.bin:
            raise RuntimeError("Install the OpenCode CLI and add it to PATH")

    def prompt(self, text, attachments=(), timeout=600):
        command = [self.bin, "run", "--pure", "--format", "json", "--model", "vision/" + MODEL]
        if not self.session:
            command.extend(["--title", "Vision benchmark"])
        if self.session:
            command.extend(["--session", self.session])
        for attachment in attachments:
            command.extend(["--file", str(attachment)])
        result = subprocess.run(
            [*command, "--", text], cwd=self.directory, env=self.env,
            capture_output=True, text=True, timeout=timeout,
        )
        response = []
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            self.session = event.get("sessionID") or self.session
            if event.get("type") == "error":
                raise RuntimeError("OpenCode reported an error: " + str(event.get("error", {}).get("name") or event.get("error", {}).get("_tag", "unknown")))
            if event.get("type") == "text":
                response.append(event["part"]["text"])
        if result.returncode:
            raise RuntimeError(f"OpenCode exited with status {result.returncode}")
        if not self.session:
            raise RuntimeError("OpenCode did not create a session")
        return "\n".join(response)
