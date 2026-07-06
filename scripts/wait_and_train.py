"""Orchestrator: poll dspark-gen until idle, then auto-run YOLOv8 training.

Polls the existing Colab T4 session every 60s. When it transitions from BUSY
to IDLE, restarts the kernel (to free GPU memory held by the prior LLM job),
executes train_colab.py on the T4, then downloads weights + metrics locally.

Run from project root:
    python scripts/wait_and_train.py
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

SESSION = "dspark-gen"
TRAIN_SCRIPT = "scripts/train_colab.py"
POLL_INTERVAL = 60       # seconds between status checks
MAX_WAIT = 3300          # 55 min cap (stays under the 3600s bash limit)
TRAIN_TIMEOUT = 1200     # 20 min for 25-epoch training on T4
DOWNLOAD_TIMEOUT = 120   # 2 min for file downloads


def run_cmd(cmd, timeout=30):
    """Run a command, return (exit_code, combined_output)."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT after {timeout}s"
    except Exception as e:
        return -1, str(e)


def get_status():
    """Return session status string (BUSY / IDLE / UNKNOWN)."""
    code, out = run_cmd(["colab", "status", "-s", SESSION], timeout=30)
    # Output line looks like:
    # [dspark-gen] gpu-t4-... | Hardware: T4 | Variant: GPU | Status: BUSY (exec(...))
    m = re.search(r"Status:\s*(\w+)", out)
    return m.group(1) if m else "UNKNOWN"


def main():
    # cd to project root (parent of scripts/)
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    print(f"[orchestrator] Working dir: {project_root}", flush=True)
    print(f"[orchestrator] Polling '{SESSION}' every {POLL_INTERVAL}s "
          f"(max wait {MAX_WAIT}s)", flush=True)

    t0 = time.time()
    while time.time() - t0 < MAX_WAIT:
        status = get_status()
        elapsed = int(time.time() - t0)
        print(f"  [{elapsed:>5d}s] {SESSION} status: {status}", flush=True)

        if status and "BUSY" not in status.upper():
            # ── Session is idle — train ────────────────────────────────────
            print(f"\n[orchestrator] {SESSION} is {status}. "
                  f"Restarting kernel to clear GPU memory...", flush=True)
            rc, out = run_cmd(
                ["colab", "restart-kernel", "-s", SESSION], timeout=30
            )
            print(f"  restart-kernel exit: {rc}", flush=True)
            time.sleep(5)

            # ── Run training ───────────────────────────────────────────────
            print(f"\n[orchestrator] Launching YOLOv8 training "
                  f"(25 epochs, timeout {TRAIN_TIMEOUT}s)...", flush=True)
            rc, out = run_cmd(
                ["colab", "exec", "-s", SESSION, "-f", TRAIN_SCRIPT,
                 "--timeout", str(TRAIN_TIMEOUT)],
                timeout=TRAIN_TIMEOUT + 60,
            )
            print(out, flush=True)
            print(f"\n[orchestrator] colab exec exit code: {rc}", flush=True)

            if rc == 0:
                # ── Download artifacts ─────────────────────────────────────
                print("\n[orchestrator] Downloading trained weights "
                      "and metrics...", flush=True)
                os.makedirs("weights", exist_ok=True)
                os.makedirs("runs", exist_ok=True)

                rc1, out1 = run_cmd(
                    ["colab", "download", "-s", SESSION,
                     "/content/runs/detect/train/weights/best.pt",
                     "weights/best.pt"],
                    timeout=DOWNLOAD_TIMEOUT,
                )
                print(f"  best.pt download: {rc1}", flush=True)

                rc2, out2 = run_cmd(
                    ["colab", "download", "-s", SESSION,
                     "/content/metrics.json", "runs/metrics.json"],
                    timeout=DOWNLOAD_TIMEOUT,
                )
                print(f"  metrics.json download: {rc2}", flush=True)

                if rc1 == 0 and rc2 == 0:
                    print("\n[orchestrator] SUCCESS. Artifacts saved:",
                          flush=True)
                    print(f"  weights/best.pt", flush=True)
                    print(f"  runs/metrics.json", flush=True)
                    return 0
                else:
                    print("\n[orchestrator] Training succeeded but "
                          "download failed. Manual recovery:", flush=True)
                    print(f"  colab download -s {SESSION} "
                          "/content/runs/detect/train/weights/best.pt "
                          "weights/best.pt", flush=True)
                    print(f"  colab download -s {SESSION} "
                          "/content/metrics.json runs/metrics.json",
                          flush=True)
                    return 2
            else:
                print(f"\n[orchestrator] Training FAILED (exit {rc}). "
                      f"See output above for diagnostics.", flush=True)
                return rc

        time.sleep(POLL_INTERVAL)

    # ── Timeout ────────────────────────────────────────────────────────────
    elapsed = int(time.time() - t0)
    print(f"\n[orchestrator] Timed out after {elapsed}s. "
          f"'{SESSION}' is still BUSY.", flush=True)
    print("Re-run this script to continue waiting, or stop the session "
          "manually: colab stop -s " + SESSION, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
