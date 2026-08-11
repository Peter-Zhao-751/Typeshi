"""Whole-run dashboard. Leave it running in its own terminal tab:

    uv run python scripts/watch_run.py

One screen for everything the Phase-1 run has in flight: GPU, the training
run (same parsing as watch_training.py, which stays the training-only view),
dataset builds, the eval, and any other logged background job — with a loud
line for anything that died. Refreshes every few seconds; --once renders a
single frame.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from watch_training import LOSS_RE, STEP_RE, sparkline  # same scripts/ dir

REFRESH_S = 5
FAIL_MARKS = ("Traceback", "out of memory", "OutOfMemory", "Killed", "FATAL")

# label -> log glob; newest matching file is shown. Anything failing shows red
# regardless of which panel it belongs to.
JOBS = {
    "dataset build": "build-*.log",
    "eval": "eval*.log",
    "hf download": "hf-download*.log",
    "kernel build": "build-causal-conv1d.log",
}


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def alive(pattern: str) -> bool:
    return bool(sh("pgrep", "-f", pattern))


def newest(glob: str) -> Path | None:
    hits = sorted(Path("logs").glob(glob), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def last_line(path: Path) -> str:
    # \r-progress bars make one enormous physical line; take the last segment.
    tail = path.read_text(errors="replace")[-4000:]
    parts = [s.strip() for s in re.split(r"[\r\n]", tail) if s.strip()]
    return parts[-1] if parts else ""


def age(path: Path) -> str:
    s = time.time() - path.stat().st_mtime
    return f"{s:.0f}s" if s < 120 else f"{s/60:.0f}m"


def gpu_panel() -> list[str]:
    q = sh("nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw",
           "--format=csv,noheader,nounits")
    if not q:
        return ["  (no GPU visible)"]
    name, util, used, total, power = [x.strip() for x in q.split(",")]
    filled = int(int(util) / 100 * 20)
    return [f"  {name}   [{'█' * filled}{'░' * (20 - filled)}] {util}%   "
            f"{int(used)/1024:.0f}/{int(total)/1024:.0f} GB   {float(power):.0f} W"]


def training_panel() -> list[str]:
    log = newest("train-*.log")
    if log is None:
        return ["  (no logs/train-*.log yet)"]
    raw = log.read_text(errors="replace").replace("\r", "\n")
    running = alive("typeshi.train_motor")
    done = "train_runtime" in raw
    state = "🟢 running" if running else ("✅ complete" if done else "💀 DIED")
    lines = [f"  {log.name}   {state}   (log {age(log)} old)"]
    if steps := STEP_RE.findall(raw):
        pct, step, total, elapsed, eta, rate = steps[-1]
        filled = int(int(pct) / 100 * 26)
        lines.append(f"  [{'█' * filled}{'░' * (26 - filled)}] {pct}%  "
                     f"step {step}/{total}  eta {eta}  {rate}s/step")
    if losses := LOSS_RE.findall(raw):
        vals = [float(l) for l, _, _, _ in losses]
        _, ent, acc, epoch = losses[-1]
        lines.append(f"  loss {vals[0]:.2f} → {vals[-1]:.3f}  {sparkline(vals, 30)}")
        lines.append(f"  epoch {float(epoch):.2f}  entropy {float(ent):.2f}  "
                     f"token-acc {float(acc)*100:.1f}%")
    return lines


def jobs_panel() -> list[str]:
    lines = []
    for label, glob in JOBS.items():
        log = newest(glob)
        if log is None or "train" in log.name:
            continue
        tail = last_line(log)[:56]
        bad = any(m in log.read_text(errors="replace")[-8000:] for m in FAIL_MARKS)
        mark = "❌" if bad else "  "
        lines.append(f"{mark}{label:>14}: {tail}   ({age(log)})")
    return lines or ["  (nothing logged)"]


def report_panel() -> list[str]:
    rep = Path("eval_report.json")
    if not rep.exists():
        return []
    import json
    try:
        r = json.loads(rep.read_text())
    except json.JSONDecodeError:
        return [f"  eval_report.json: unreadable ({age(rep)})"]
    gates = {k: v for k, v in r.items() if k.startswith("pass_")}
    met = r.get("tier1_met")
    lines = [f"  eval_report.json ({age(rep)} old)   tier1_met: "
             + ("✅ TRUE" if met else f"❌ {met}")]
    lines += [f"    {'✅' if v else '❌'} {k}" for k, v in gates.items()]
    return lines


def render() -> None:
    print("\x1b[2J\x1b[H", end="")
    print(f"┏━━ TYPESHI RUN ━━ {datetime.now():%H:%M:%S} ━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    for title, panel in (("gpu", gpu_panel()), ("training", training_panel()),
                         ("jobs", jobs_panel()), ("tier-1", report_panel())):
        if not panel:
            continue
        print(f"┃ {title}")
        for line in panel:
            print(f"┃ {line}")
        print("┃")
    print("┗" + "━" * 61)
    print(f"  refreshing every {REFRESH_S}s — ctrl-c to quit")


def main() -> None:
    while True:
        render()
        if "--once" in sys.argv:
            return
        time.sleep(REFRESH_S)


if __name__ == "__main__":
    main()
