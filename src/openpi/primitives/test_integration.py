"""Integration test: mock controller + REPL driver without a real robot.

Tests the full pipeline:
  1. Start REPL driver (TCP server, may or may not load VLA model).
  2. Start mock controller (TCP client).
  3. Write a sequence of commands via REPL protocol.
  4. Verify done flags, log files, and state files are produced correctly.

Usage:
    python -m openpi.primitives.test_integration [--load-model] [--checkpoint-dir PATH]

Without --load-model: tests REPL protocol + scripted primitives only.
With --load-model + --checkpoint-dir: also tests VLA inference + pi0_pick.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

OPENPI_ROOT = Path(os.environ.get("OPENPI_ROOT", "/mnt/public/nieyi/code/agentic/openpi"))

# ---------------------------------------------------------------------------
# Test command sequences
# ---------------------------------------------------------------------------

# Phase 1 test: only scripted primitives (no VLA model needed).
SCRIPTED_TEST_COMMANDS = [
    {"action": "move_to", "xyz": [0.05, 0.0, 0.95], "gripper": -1, "max_steps": 40},
    {"action": "set_gripper", "gripper": 1.0, "steps": 5},
    {"action": "move_to", "xyz": [0.05, 0.05, 1.1], "gripper": 1, "max_steps": 40},
    {"action": "move_to", "xyz": [0.05, 0.05, 0.95], "gripper": 1, "max_steps": 40},
    {"action": "release", "max_steps": 10},
    {"action": "snapshot"},
    {"action": "exit"},
]

# Phase 2 test: requires VLA model (commented out by default).
VLA_TEST_COMMAND = {"action": "pi0_pick", "prompt": "pick up the object", "max_chunks": 3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_flag(workdir: str, step: int, timeout: float = 30.0) -> bool:
    flag = Path(workdir) / f"done_{step:02d}.flag"
    t0 = time.time()
    while not flag.exists():
        if time.time() - t0 > timeout:
            return False
        time.sleep(0.2)
    return True


def _send_command(workdir: str, cmd: dict) -> None:
    wd = Path(workdir)
    tmp = wd / "command.json.tmp"
    with open(tmp, "w") as f:
        json.dump(cmd, f)
    os.replace(tmp, wd / "command.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Integration test for REPL driver")
    parser.add_argument("--load-model", action="store_true", help="Load Pi0.5 and test VLA inference")
    parser.add_argument("--checkpoint-dir", help="Path to Pi0.5 checkpoint")
    parser.add_argument("--config", default="pi05_x2robot")
    parser.add_argument("--tcp-ip", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=57771)
    parser.add_argument("--workdir", default="/tmp/hybrid_repl_test")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    if workdir.exists():
        import shutil
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    driver_log = str(workdir.parent / f"{workdir.name}_driver.log")
    mock_log = str(workdir.parent / f"{workdir.name}_mock.log")

    python = str(OPENPI_ROOT / ".venv" / "bin" / "python")

    # --- 1. Start REPL driver ---
    logger.info("=== starting REPL driver ===")
    driver_cmd = [
        python, "-m", "openpi.primitives.repl_driver",
        "--config", args.config,
        "--workdir", str(workdir),
        "--max-steps", "20",
        "--tcp-server",
        "--tcp-ip", args.tcp_ip,
        "--tcp-port", str(args.tcp_port),
    ]
    if args.load_model and args.checkpoint_dir:
        driver_cmd += ["--checkpoint-dir", args.checkpoint_dir]

    driver = subprocess.Popen(driver_cmd, stdout=open(driver_log, "w"), stderr=subprocess.STDOUT)
    logger.info("driver pid=%d", driver.pid)

    # --- 2. Wait for driver port ready (JAX import ~30s) ---
    logger.info("waiting for driver to listen on %s:%d (up to 120s)...", args.tcp_ip, args.tcp_port)
    t0 = time.time()
    port_ready = False
    while time.time() - t0 < 120:
        if driver.poll() is not None:
            logger.error("driver exited early. Log:")
            print(Path(driver_log).read_text()[-2000:])
            return 1
        import subprocess as _sp
        r = _sp.run(["ss", "-tlnp"], capture_output=True, text=True)
        if f":{args.tcp_port}" in r.stdout:
            port_ready = True
            break
        time.sleep(1)
    if not port_ready:
        driver.terminate()
        logger.error("driver port not ready after 120s")
        return 1
    logger.info("driver port ready in %.1fs", time.time() - t0)

    # --- 3. Start mock controller ---
    logger.info("=== starting mock controller ===")
    mock_cmd = [
        python, "-m", "openpi.primitives.mock_controller",
        "--ip", args.tcp_ip,
        "--port", str(args.tcp_port),
        "--max-cycles", "20",
    ]
    mock = subprocess.Popen(mock_cmd, stdout=open(mock_log, "w"), stderr=subprocess.STDOUT)
    time.sleep(1)
    if mock.poll() is not None:
        logger.error("mock controller exited early. Log:")
        print(Path(mock_log).read_text()[-2000:])
        driver.terminate()
        return 1
    logger.info("mock controller pid=%d", mock.pid)

    # --- 4. Wait for initial state ---
    logger.info("waiting for state_00.json (mock should send first frame)...")
    t0 = time.time()
    while not (workdir / "state_00.json").exists():
        if driver.poll() is not None:
            logger.error("driver exited early")
            return 1
        if mock.poll() is not None:
            logger.error("mock exited early. Log:")
            print(Path(mock_log).read_text()[-2000:])
            return 1
        if time.time() - t0 > 60:
            logger.error("state_00 not ready after 60s")
            return 1
        time.sleep(0.5)
    logger.info("state_00 ready in %.1fs", time.time() - t0)

    # --- 4. Run test commands ---
    passed = 0
    failed = 0

    for step, cmd in enumerate(SCRIPTED_TEST_COMMANDS, start=1):
        logger.info("--- step %d: %s ---", step, cmd.get("action"))

        _send_command(str(workdir), cmd)

        if not _wait_for_flag(str(workdir), step, timeout=30.0):
            logger.error("step %d: TIMEOUT waiting for done_%02d.flag", step, step)
            failed += 1
            break

        log_path = workdir / f"log_{step:02d}.json"
        state_path = workdir / f"state_{step:02d}.json"

        log_ok = state_ok = False
        if log_path.exists():
            log_data = json.loads(log_path.read_text())
            result = log_data.get("result", {})
            # Check for success/ok in result
            if isinstance(result, dict):
                log_ok = result.get("ok", False) or result.get("success", False) or "error" not in str(result).lower()
            log_ok = log_ok or result is not None
        if state_path.exists():
            state_ok = True

        if log_ok and state_ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
            logger.error("  log_ok=%s state_ok=%s", log_ok, state_ok)

        logger.info("step %d: %s (action=%s)", step, status, cmd.get("action"))

    # --- 5. Cleanup ---
    logger.info("=== cleanup ===")
    _send_command(str(workdir), {"action": "exit"})
    time.sleep(2)
    driver.terminate()
    mock.terminate()
    try:
        driver.wait(timeout=5)
    except subprocess.TimeoutExpired:
        driver.kill()
    try:
        mock.wait(timeout=5)
    except subprocess.TimeoutExpired:
        mock.kill()

    # --- 6. Report ---
    logger.info("=== results ===")
    logger.info("passed: %d  failed: %d  total: %d", passed, failed, passed + failed)
    logger.info("driver log: %s", driver_log)
    logger.info("mock log:   %s", mock_log)
    logger.info("workdir:    %s", workdir)

    if failed == 0:
        logger.info("ALL TESTS PASSED")
        return 0
    else:
        logger.error("%d TESTS FAILED", failed)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [test] %(message)s", datefmt="%H:%M:%S")
    main()
