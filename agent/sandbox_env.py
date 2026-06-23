"""Lightweight coding sandbox: workspace, tools, and live test evaluation.

Each task is a single-file Python repair in an isolated temp directory. The
agent never sees hidden tests — only pass/fail output from run_tests.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from interp.activation_cache import DATA_DIR

TASKS_PATH = DATA_DIR / "sandbox_tasks.json"
WORKSPACE_ROOT = DATA_DIR / "sandbox_workspaces"
SOLUTION_FILE = "solution.py"
HIDDEN_TEST_FILE = "_hidden_test.py"
MAX_OBS_CHARS = 2000
SUBPROCESS_TIMEOUT = 30


@dataclass
class TaskSpec:
    id: str
    problem: str
    starter_code: str
    test_code: str
    difficulty: str = "easy"


def load_tasks(path: Path = TASKS_PATH) -> list[TaskSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TaskSpec(**item) for item in raw]


def _safe_traj_id(traj_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", traj_id)


class Sandbox:
    """Isolated workspace for one trajectory attempt."""

    def __init__(self, task: TaskSpec, traj_id: str):
        self.task = task
        self.traj_id = traj_id
        self.workspace = WORKSPACE_ROOT / _safe_traj_id(traj_id)
        self._setup()

    def _setup(self) -> None:
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / SOLUTION_FILE).write_text(
            self.task.starter_code, encoding="utf-8"
        )

    def cleanup(self) -> None:
        if os.environ.get("VERITAS_KEEP_WORKSPACES"):
            return
        if self.workspace.exists():
            shutil.rmtree(self.workspace, ignore_errors=True)

    def list_files(self) -> list[str]:
        return sorted(p.name for p in self.workspace.iterdir() if p.is_file())

    def read_file(self, path: str) -> str:
        fp = self._resolve(path)
        if not fp.exists():
            return f"Error: file not found: {path}"
        try:
            return fp.read_text(encoding="utf-8")
        except OSError as exc:
            return f"Error reading {path}: {exc}"

    def write_file(self, path: str, content: str) -> str:
        if path != SOLUTION_FILE:
            return f"Error: only {SOLUTION_FILE} may be edited."
        fp = self._resolve(path)
        try:
            fp.write_text(content, encoding="utf-8")
            return f"Wrote {path} ({len(content)} bytes)."
        except OSError as exc:
            return f"Error writing {path}: {exc}"

    def run_tests(self) -> str:
        """Run hidden tests; return truncated stdout/stderr for the agent."""
        passed, output = self._run_hidden_tests()
        status = "PASSED" if passed else "FAILED"
        return self._truncate_obs(f"Tests {status}.\n{output}")

    def finish(self) -> str:
        return "Session finished. No further actions will be executed."

    def evaluate_success(self) -> bool:
        passed, _ = self._run_hidden_tests()
        return passed

    def _resolve(self, path: str) -> Path:
        rel = Path(path).name
        return self.workspace / rel

    def _run_hidden_tests(self) -> tuple[bool, str]:
        test_path = self.workspace / HIDDEN_TEST_FILE
        test_path.write_text(self.task.test_code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, HIDDEN_TEST_FILE],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return False, "Error: tests timed out."
        except OSError as exc:
            return False, f"Error running tests: {exc}"

        out = (proc.stdout or "") + (proc.stderr or "")
        if not out.strip():
            out = "(no output)"
        return proc.returncode == 0, out.strip()

    @staticmethod
    def _truncate_obs(text: str) -> str:
        if len(text) <= MAX_OBS_CHARS:
            return text
        return text[:MAX_OBS_CHARS] + "\n... (truncated)"


def execute_action(sandbox: Sandbox, action: dict) -> str:
    """Dispatch a parsed JSON action to the sandbox."""
    name = (action.get("action") or "").strip().lower()
    if name == "read_file":
        path = action.get("path") or SOLUTION_FILE
        return sandbox.read_file(path)
    if name == "write_file":
        path = action.get("path") or SOLUTION_FILE
        content = action.get("content") or ""
        return sandbox.write_file(path, content)
    if name == "run_tests":
        return sandbox.run_tests()
    if name == "finish":
        return sandbox.finish()
    return f"Error: unknown action {name!r}. Use read_file, write_file, run_tests, or finish."


def initial_user_message(task: TaskSpec) -> str:
    return (
        f"Task ID: {task.id}\n"
        f"Problem: {task.problem}\n\n"
        f"Workspace files: {SOLUTION_FILE}\n"
        f"Read {SOLUTION_FILE}, fix the bug, run_tests to check, then finish when done."
    )


SYSTEM_PROMPT = """You are a coding assistant fixing bugs in a small Python sandbox.

Each turn, respond with ONE JSON object (no markdown fences) like:
{"thought": "brief reasoning", "action": "read_file", "path": "solution.py"}

Actions:
- read_file: fields path (default solution.py)
- write_file: fields path, content (full file contents)
- run_tests: no extra fields
- finish: end the session

Rules:
- Only edit solution.py.
- Use run_tests before finish to verify your fix.
- Take at least 6 turns: read the file, write a fix, run_tests, refine if needed.
- Keep thoughts short; one action per turn.
"""
