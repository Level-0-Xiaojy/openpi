"""Hybrid LLM-in-the-loop agent runner for ARX real robot.

Drives repl_driver.py through LLM tool calls. Supports:
- Anthropic (Claude) via anthropic SDK
- DeepSeek via OpenAI-compatible SDK
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

OPENPI_ROOT = Path(os.environ.get("OPENPI_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
DEFAULT_WORKDIR = "/tmp/hybrid_repl"

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import tools as tools_mod
from tools import TOOLS_SPEC, execute_tool, tool_result_to_content_blocks
from prompts import SYSTEM_PROMPT, INITIAL_USER_TEMPLATE


# ---------------------------------------------------------------------------
# Driver lifecycle
# ---------------------------------------------------------------------------

DRIVER_SCRIPT = str(OPENPI_ROOT / "src" / "openpi" / "primitives" / "repl_driver.py")


def start_driver(
    workdir: str = DEFAULT_WORKDIR,
    config: str = "pi0_x2robot",
    checkpoint_dir: str | None = None,
    max_steps: int = 40,
    tcp_server: bool = False,
    tcp_ip: str = "192.168.77.58",
    tcp_port: int = 57770,
    log_path: str | None = None,
    ready_timeout_s: float = 300.0,
) -> subprocess.Popen:
    wd = Path(workdir)
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        log_path = str(wd.parent / f"{wd.name}_driver.log")
    python_bin = str(OPENPI_ROOT / ".venv" / "bin" / "python")
    cmd = [
        python_bin, "-m", "openpi.primitives.repl_driver",
        "--config", config, "--workdir", str(wd), "--max-steps", str(max_steps),
        "--tcp-ip", tcp_ip, "--tcp-port", str(tcp_port),
    ]
    if checkpoint_dir:
        cmd += ["--checkpoint-dir", checkpoint_dir]
    if tcp_server:
        cmd.append("--tcp-server")
    print(f"[agent] driver cmd: {' '.join(cmd)}")
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(OPENPI_ROOT))
    print(f"[agent] waiting for state_00.json (model load ~90s)...")
    t0 = time.time()
    while not (wd / "state_00.json").exists():
        time.sleep(2)
        if proc.poll() is not None:
            print(f"[agent] driver EXITED before ready.")
            print(Path(log_path).read_text()[-2000:])
            raise RuntimeError("driver exited prematurely")
        if time.time() - t0 > ready_timeout_s:
            proc.terminate()
            raise RuntimeError(f"driver not ready after {ready_timeout_s}s")
    print(f"[agent] driver ready in {time.time() - t0:.1f}s")
    return proc


def stop_driver(proc: subprocess.Popen, workdir: str = DEFAULT_WORKDIR, timeout: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    cmd_path = Path(workdir) / "command.json"
    try:
        with open(cmd_path, "w") as f:
            json.dump({"action": "exit"}, f)
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# Anthropic <-> OpenAI tool format conversion
# ---------------------------------------------------------------------------

def _anthropic_tools_to_openai(anthropic_tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool spec to OpenAI function spec."""
    openai_tools = []
    for t in anthropic_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        })
    return openai_tools


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _short_repr(obj, maxlen=200):
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return s if len(s) <= maxlen else s[:maxlen] + "...(+%d)" % (len(s) - maxlen)


def _serialize_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            out.append({"role": m["role"], "content": c})
            continue
        new_blocks = []
        for b in c:
            if isinstance(b, dict):
                if b.get("type") == "image":
                    new_blocks.append({"type": "image", "source": {"_omitted_for_transcript": True}})
                else:
                    new_blocks.append(b)
                continue
            bd: dict = {"type": getattr(b, "type", "?")}
            for attr in ("text", "name", "input", "id"):
                if hasattr(b, attr):
                    bd[attr] = getattr(b, attr)
            new_blocks.append(bd)
        out.append({"role": m["role"], "content": new_blocks})
    return out


# ---------------------------------------------------------------------------
# Anthropic agent loop
# ---------------------------------------------------------------------------

def _run_anthropic_loop(client, model, system, messages, max_turns, max_tokens, verbose):
    import anthropic
    finish_result = None
    total_in = total_out = 0
    n_tool_calls = 0

    for turn in range(1, max_turns + 1):
        if verbose:
            print(f"\n[agent] === turn {turn}/{max_turns} ===")
        response = None
        for outer in range(3):
            try:
                response = client.messages.create(
                    model=model, max_tokens=max_tokens, system=system,
                    tools=TOOLS_SPEC, messages=messages,
                )
                break
            except (anthropic.APIConnectionError, anthropic.APITimeoutError,
                    anthropic.InternalServerError, anthropic.RateLimitError) as e:
                wait = 10 * (outer + 1)
                if verbose:
                    print(f"[agent] API error '{type(e).__name__}' — sleeping {wait}s")
                time.sleep(wait)
        if response is None:
            if verbose:
                print("[agent] giving up after 3 retries")
            break

        u = response.usage
        total_in += u.input_tokens
        total_out += u.output_tokens

        if verbose:
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"[claude] {block.text.strip()}")
                elif block.type == "tool_use":
                    print(f"[tool→] {block.name}({_short_repr(block.input, 250)})")
            print(f"[usage] in={u.input_tokens} out={u.output_tokens} "
                  f"stop={response.stop_reason}")

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "tool_use":
            finish_result, n_tool_calls = _process_anthropic_tools(response, messages, n_tool_calls, verbose)
            if finish_result:
                break
        elif response.stop_reason == "end_turn":
            if verbose:
                print("[agent] end_turn. Stopping.")
            break
        else:
            if verbose:
                print(f"[agent] unexpected stop: {response.stop_reason}")
            break

    stats = {"total_input_tokens": total_in, "total_output_tokens": total_out,
             "turns_used": turn, "tool_calls": n_tool_calls}
    return finish_result, messages, stats


def _process_anthropic_tools(response, messages, n_tool_calls, verbose):
    tool_results = []
    finish_result = None
    for block in response.content:
        if block.type != "tool_use":
            continue
        n_tool_calls += 1
        result = execute_tool(block.name, block.input or {})
        if isinstance(result, dict) and result.get("_finish"):
            finish_result = result
        if verbose:
            summary = {k: v for k, v in result.items()
                       if k not in ("state", "content", "log", "_image_path")} \
                if isinstance(result, dict) else result
            print(f"[tool←] {block.name}: {_short_repr(summary, 350)}")
        tool_results.append({
            "type": "tool_result", "tool_use_id": block.id,
            "content": tool_result_to_content_blocks(result),
        })
    messages.append({"role": "user", "content": tool_results})
    return finish_result, n_tool_calls


# ---------------------------------------------------------------------------
# DeepSeek (OpenAI-compatible) agent loop
# ---------------------------------------------------------------------------

def _run_deepseek_loop(client, model, system, messages, max_turns, max_tokens, verbose):
    openai_tools = _anthropic_tools_to_openai(TOOLS_SPEC)
    finish_result = None
    total_in = total_out = 0
    n_tool_calls = 0

    # Prepend system message
    ds_messages = [{"role": "system", "content": system}] + messages

    for turn in range(1, max_turns + 1):
        if verbose:
            print(f"\n[agent] === turn {turn}/{max_turns} ===")
        response = None
        for outer in range(3):
            try:
                response = client.chat.completions.create(
                    model=model, max_tokens=max_tokens,
                    tools=openai_tools, messages=ds_messages,
                )
                break
            except Exception as e:
                wait = 10 * (outer + 1)
                if verbose:
                    print(f"[agent] API error '{type(e).__name__}: {e}' — sleeping {wait}s")
                time.sleep(wait)
        if response is None:
            if verbose:
                print("[agent] giving up after 3 retries")
            break

        choice = response.choices[0]
        msg = choice.message
        u = response.usage
        total_in += u.prompt_tokens
        total_out += u.completion_tokens

        if verbose and msg.content:
            print(f"[ds] {msg.content.strip()}")
        if verbose and msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                print(f"[tool→] {tc.function.name}({_short_repr(args, 250)})")
        if verbose:
            print(f"[usage] in={u.prompt_tokens} out={u.completion_tokens} "
                  f"finish={choice.finish_reason}")

        ds_messages.append({"role": "assistant", "content": msg.content,
                            "tool_calls": [tc.model_dump() for tc in msg.tool_calls] if msg.tool_calls else None})
        # Clean up: remove None fields
        ds_messages[-1] = {k: v for k, v in ds_messages[-1].items() if v is not None}

        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            tool_results = []
            for tc in msg.tool_calls:
                n_tool_calls += 1
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = execute_tool(tc.function.name, args)
                if isinstance(result, dict) and result.get("_finish"):
                    finish_result = result
                if verbose:
                    summary = {k: v for k, v in result.items()
                               if k not in ("state", "content", "log", "_image_path")} \
                        if isinstance(result, dict) else result
                    print(f"[tool←] {tc.function.name}: {_short_repr(summary, 350)}")
                blocks = tool_result_to_content_blocks(result)
                text_blocks = [b for b in blocks if b.get("type") == "text"] if isinstance(blocks, list) else []
                content_str = "\n".join(b.get("text", "") for b in text_blocks)
                if not content_str:
                    content_str = json.dumps(result, default=str)
                tool_results.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": content_str,
                })
            ds_messages.extend(tool_results)

            if finish_result is not None:
                if verbose:
                    print(f"\n[agent] FINISH: {finish_result}")
                break
        elif choice.finish_reason == "stop":
            if verbose:
                print("[agent] model ended. Stopping.")
            break
        else:
            if verbose:
                print(f"[agent] unexpected finish: {choice.finish_reason}")
            break

    stats = {"total_input_tokens": total_in, "total_output_tokens": total_out,
             "turns_used": turn, "tool_calls": n_tool_calls}
    return finish_result, ds_messages, stats


# ---------------------------------------------------------------------------
# High-level entrypoint
# ---------------------------------------------------------------------------

def run_one_cell(
    experiment_name: str,
    api_key: str,
    model: str = "claude-sonnet-4-5",
    provider: str = "anthropic",
    max_turns: int = 80,
    max_tokens: int = 4096,
    output_dir: str | None = None,
    no_driver: bool = False,
    verbose: bool = True,
    base_url: str | None = None,
    workdir: str = DEFAULT_WORKDIR,
    config: str = "pi0_x2robot",
    checkpoint_dir: str | None = None,
    tcp_server: bool = False,
    tcp_ip: str = "192.168.77.58",
    tcp_port: int = 57770,
) -> dict:
    if output_dir is None:
        output_dir = str(OPENPI_ROOT / "workspace" / "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tools_mod.set_workdir(workdir)

    user_msg = INITIAL_USER_TEMPLATE.format(
        experiment_name=experiment_name,
        output_dir=output_dir, workdir=workdir,
    )

    proc = None
    if not no_driver:
        proc = start_driver(
            workdir=workdir, config=config, checkpoint_dir=checkpoint_dir,
            tcp_server=tcp_server, tcp_ip=tcp_ip, tcp_port=tcp_port,
        )
    else:
        if not (Path(workdir) / "state_00.json").exists():
            raise RuntimeError(f"--no_driver but {workdir}/state_00.json missing")

    def _emergency_save(error_msg: str | None = None):
        """Salvage recipe + audit from workdir if agent crashes mid-run."""
        wd = Path(workdir)
        out = Path(output_dir)
        if not wd.exists():
            return
        recipe_path = out / f"recipe_{experiment_name}.jsonl"
        audit_path = out / f"{experiment_name}.json"
        if recipe_path.exists() and audit_path.exists():
            return

        logs = {}
        for lp in sorted(wd.glob("log_*.json")):
            try:
                ln = int(lp.stem.split("_")[1])
                logs[ln] = json.loads(lp.read_text())
            except Exception:
                continue

        if not recipe_path.exists() and logs:
            lines = []
            for ln in sorted(logs.keys()):
                cmd = logs[ln].get("command") or {}
                if cmd.get("action") in ("exit",):
                    continue
                lines.append(json.dumps(cmd))
            if lines:
                out.mkdir(parents=True, exist_ok=True)
                recipe_path.write_text("\n".join(lines) + "\n")
                if verbose:
                    print(f"[agent] [emergency_save] wrote {recipe_path}")

        if not audit_path.exists() and logs:
            last_log = logs[max(logs)]
            audit = {
                "experiment": experiment_name,
                "model": model,
                "provider": provider,
                "regime": "strict",
                "strategy_notes": f"emergency-saved after agent error: {error_msg}" if error_msg else "emergency-saved (agent did not call finish)",
                "steps_completed": len(logs),
                "agent_error": error_msg,
            }
            out.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps(audit, indent=2, default=str))
            if verbose:
                print(f"[agent] [emergency_save] wrote {audit_path}")

    client_kwargs = {"api_key": api_key, "max_retries": 8, "timeout": 120.0}
    if base_url:
        client_kwargs["base_url"] = base_url

    t0 = time.time()
    agent_error = None

    try:
        if provider == "deepseek":
            import openai
            ds_kwargs = {"api_key": api_key, "max_retries": 8, "timeout": 120.0}
            if base_url:
                ds_kwargs["base_url"] = base_url
            client = openai.OpenAI(**ds_kwargs)
            messages = [{"role": "user", "content": user_msg}]
            finish_result, messages, stats = _run_deepseek_loop(
                client, model, SYSTEM_PROMPT, messages,
                max_turns=max_turns, max_tokens=max_tokens, verbose=verbose,
            )
        else:
            import anthropic
            if base_url:
                client_kwargs["base_url"] = base_url
            client = anthropic.Anthropic(**client_kwargs)
            finish_result, messages, stats = _run_anthropic_loop(
                client, model, SYSTEM_PROMPT, messages,
                max_turns=max_turns, max_tokens=max_tokens, verbose=verbose,
            )
    except Exception as e:
        agent_error = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"[agent] EXCEPTION: {agent_error}")
    finally:
        try:
            _emergency_save(agent_error)
        except Exception:
            pass
        if proc is not None:
            stop_driver(proc, workdir=workdir)

    elapsed = time.time() - t0
    transcript_path = Path(output_dir) / f"transcript_{experiment_name}.json"
    record = {
        "experiment": experiment_name, "model": model, "provider": provider,
        "elapsed_s": round(elapsed, 1), "finish": finish_result,
        "stats": stats, "agent_error": agent_error,
        "messages": _serialize_messages(messages),
    }
    with open(transcript_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    if verbose:
        print(f"\n[agent] elapsed: {elapsed:.1f}s")
        print(f"[agent] transcript: {transcript_path}")
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Hybrid LLM agent for ARX real robot")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "deepseek"])
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--max_turns", type=int, default=80)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--no_driver", action="store_true")
    ap.add_argument("--workdir", default=DEFAULT_WORKDIR)
    ap.add_argument("--config", default="pi0_x2robot")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--tcp-server", action="store_true")
    ap.add_argument("--tcp-ip", default="192.168.77.58")
    ap.add_argument("--tcp-port", type=int, default=57770)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: set DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY env var or pass --api_key")
        return 2
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL")

    run_one_cell(
        experiment_name=args.experiment, api_key=api_key,
        provider=args.provider, model=args.model,
        max_turns=args.max_turns, max_tokens=args.max_tokens,
        output_dir=args.output_dir, no_driver=args.no_driver,
        verbose=not args.quiet, base_url=base_url,
        workdir=args.workdir, config=args.config,
        checkpoint_dir=args.checkpoint_dir,
        tcp_server=args.tcp_server, tcp_ip=args.tcp_ip, tcp_port=args.tcp_port,
    )
    return 0


if __name__ == "__main__":
    main()
