"""Tool implementations for the real-world hybrid LLM-in-the-loop agent.

Each tool is a thin wrapper that the agent calls via Anthropic's tool-use
API. Results are JSON-serializable dicts; for image-bearing tools the
caller (runner.py) converts a `_image_path` field into a multimodal
content block.
"""

import base64
import json
import os
import re
import time
from pathlib import Path

WORKDIR = Path(os.environ.get("HYBRID_REPL_WORKDIR", "/tmp/hybrid_repl"))
OPENPI_ROOT = Path(os.environ.get("OPENPI_ROOT", "/mnt/public/nieyi/code/agentic/openpi"))


def set_workdir(path: str | os.PathLike) -> None:
    global WORKDIR
    WORKDIR = Path(path)


# ---------------------------------------------------------------------------
# Tool schema declarations
# ---------------------------------------------------------------------------

TOOLS_SPEC = [
    {
        "name": "read_text_file",
        "description": (
            "Read a UTF-8 text file. Use for calibration files "
            "(env_calibration.md), past experiment logs, and memory files. "
            "Large files are truncated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or workspace-relative path"},
                "max_chars": {"type": "integer", "description": "Max chars (default 40000)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_text_file",
        "description": (
            "Write a UTF-8 text file (creates parent dirs). Use to save "
            "experiment logs and final audit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": (
            "List files in a directory (non-recursive). Default = REPL workdir. "
            "Use to inspect the REPL working directory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Default = REPL workdir"},
            },
        },
    },
    {
        "name": "view_repl_state",
        "description": (
            "Read state_NN.json + log_NN.json + image_NN.png from the REPL "
            "workdir. If step is null, returns the latest. Returns the state "
            "JSON and embeds the camera PNG as a multimodal image content "
            "block (use this image — JSON state alone is not enough; see Rule 0)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": ["integer", "null"],
                    "description": "Step number; 0 = initial. Null = latest.",
                },
            },
        },
    },
    {
        "name": "send_command",
        "description": (
            "Write a JSON command to the REPL workdir command.json and BLOCK "
            "until the driver writes the next done_NN.flag. Returns the "
            "new state JSON + log JSON + camera image.\n\n"
            "ALLOWED actions:\n"
            "  - move_to: {action, xyz:[x,y,z], gripper:-1|+1, tol, step_clip, max_steps}\n"
            "  - release: {action, max_steps}\n"
            "  - set_gripper: {action, gripper:+1|-1, steps}\n"
            "  - snapshot: {action} — dump state only, no robot motion\n"
            "  - rotate_wrist / rotate_pitch: reorient gripper around world "
            "Z / X axis\n\n"
            "BLOCKED: exit. The runner handles driver lifecycle.\n\n"
            "  Single episode, no reset. Recover from failures within the "
            "current episode."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "object",
                    "description": "Command dict per the REPL protocol spec",
                },
                "timeout_s": {
                    "type": "number",
                    "description": "Seconds to wait for done flag (default 600)",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Declare the task finished. Call when the task goal is achieved "
            "or when genuinely stuck after honest exploration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["success", "failure", "stuck"],
                },
                "summary": {
                    "type": "string",
                    "description": "1-3 sentence summary of what worked / what failed.",
                },
            },
            "required": ["status", "summary"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = OPENPI_ROOT / p
    return p


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + f"\n\n[TRUNCATED — file is {len(text)} chars, showed first {max_chars}]"
    )


def read_text_file(path: str, max_chars: int = 40000) -> dict:
    p = _resolve(path)
    if not p.exists():
        return {"error": f"file not found: {p}"}
    if p.is_dir():
        return {"error": f"is a directory: {p}"}
    try:
        text = p.read_text(errors="replace")
    except Exception as e:
        return {"error": str(e)}
    return {"path": str(p), "size": len(text), "content": _truncate(text, max_chars)}


def write_text_file(path: str, content: str) -> dict:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"path": str(p), "bytes_written": len(content.encode("utf-8"))}


def list_dir(path: str = "") -> dict:
    p = _resolve(path) if path else WORKDIR
    if not p.exists():
        return {"error": f"directory not found: {p}"}
    files = sorted(os.listdir(p))
    return {"path": str(p), "count": len(files), "files": files}


def _latest_step() -> int | None:
    if not WORKDIR.exists():
        return None
    flag_nums = []
    for f in WORKDIR.glob("done_*.flag"):
        m = re.match(r"done_(\d+)\.flag", f.name)
        if m:
            flag_nums.append(int(m.group(1)))
    if flag_nums:
        return max(flag_nums)
    if (WORKDIR / "state_00.json").exists():
        return 0
    return None


def view_repl_state(step: int | None = None) -> dict:
    if not WORKDIR.exists():
        return {"error": "WORKDIR does not exist; driver not started"}
    if step is None:
        nn = _latest_step()
        if nn is None:
            return {"error": "no state files; driver not ready"}
    else:
        nn = step
    nn_str = f"{nn:02d}"

    state_path = WORKDIR / f"state_{nn_str}.json"
    log_path = WORKDIR / f"log_{nn_str}.json"
    image_path = WORKDIR / f"image_{nn_str}.png"

    out: dict = {"step": nn}
    if state_path.exists():
        with open(state_path) as f:
            data = json.load(f)
        out["state"] = data.get("state", data)
    else:
        out["state_error"] = f"missing {state_path}"
    if log_path.exists():
        with open(log_path) as f:
            out["log"] = json.load(f)
    if image_path.exists():
        out["_image_path"] = str(image_path)
    return out


BLOCKED_ACTIONS = {"exit"}


def send_command(command: dict, timeout_s: float = 600.0) -> dict:
    if not WORKDIR.exists():
        return {"error": "WORKDIR missing; driver not started"}

    action = command.get("action") if isinstance(command, dict) else None
    if action in BLOCKED_ACTIONS:
        return {
            "error": (
                f"action '{action}' is not available to the agent. "
                f"Recover within the current episode instead."
            ),
            "blocked_action": action,
        }

    current = _latest_step()
    if current is None:
        return {"error": "no state_00.json; driver not ready"}
    next_n = current + 1
    next_nn = f"{next_n:02d}"

    cmd_path = WORKDIR / "command.json"
    tmp_path = WORKDIR / "command.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(command, f)
    os.replace(tmp_path, cmd_path)

    flag_path = WORKDIR / f"done_{next_nn}.flag"
    t0 = time.time()
    while not flag_path.exists():
        time.sleep(0.5)
        if time.time() - t0 > timeout_s:
            return {
                "error": f"timeout after {timeout_s}s waiting for {flag_path.name}",
                "command_sent": command,
            }

    elapsed = time.time() - t0
    result = view_repl_state(next_n)
    result["agent_elapsed_s"] = round(elapsed, 1)
    return result


def finish(status: str, summary: str) -> dict:
    return {"_finish": True, "status": status, "summary": summary}


TOOL_HANDLERS = {
    "read_text_file": read_text_file,
    "write_text_file": write_text_file,
    "list_dir": list_dir,
    "view_repl_state": view_repl_state,
    "send_command": send_command,
    "finish": finish,
}


def execute_tool(name: str, input_dict: dict) -> dict:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return handler(**input_dict)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}", "got": input_dict}
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Convert tool result -> Anthropic content blocks
# ---------------------------------------------------------------------------

MAX_TEXT_BYTES_IN_RESULT = 60000


def tool_result_to_content_blocks(result):
    if not isinstance(result, dict):
        return [{"type": "text", "text": str(result)[:MAX_TEXT_BYTES_IN_RESULT]}]

    image_path = result.pop("_image_path", None)
    text = json.dumps(result, indent=2, default=str)
    if len(text) > MAX_TEXT_BYTES_IN_RESULT:
        text = text[:MAX_TEXT_BYTES_IN_RESULT] + "\n[truncated]"

    blocks = [{"type": "text", "text": text}]

    if image_path:
        p = Path(image_path)
        if p.exists():
            with open(p, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": data,
                },
            })
    return blocks
