"""Live training dashboard. Leave it running in its own terminal tab:

    uv run python scripts/watch_training.py

Tails the newest logs/train-*.log (or a path you pass), refreshing every few
seconds: step/ETA from the progress bar, loss trajectory with a sparkline,
token accuracy, and a loud banner if the process dies or the log goes stale.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REFRESH_S = 10
BARS = "▁▂▃▄▅▆▇█"
STEP_RE = re.compile(r"(\d+)%\|.*?\| (\d+)/(\d+) \[([\d:]+)<([\d:?]+), *([\d.]+)s/it\]")
LOSS_RE = re.compile(
    r"\{'loss': '([\d.]+)'.*?'entropy': '([\d.]+)'.*?"
    r"'mean_token_accuracy': '([\d.eE+-]+)', 'epoch': '([\d.]+)'\}"
)


def newest_log() -> Path:
    logs = sorted(Path("logs").glob("train-*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        sys.exit("no logs/train-*.log found")
    return logs[-1]


def sparkline(values: list[float], width: int = 40) -> str:
    if len(values) < 2:
        return ""
    values = values[-width:]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(BARS[int((v - lo) / span * (len(BARS) - 1))] for v in values)


def render(path: Path) -> None:
    raw = path.read_text(errors="replace").replace("\r", "\n")
    losses = LOSS_RE.findall(raw)
    steps = STEP_RE.findall(raw)
    alive = bool(subprocess.run(
        ["pgrep", "-f", "typeshi.train_motor"], capture_output=True
    ).stdout.strip())
    stale_s = time.time() - path.stat().st_mtime

    print("\x1b[2J\x1b[H", end="")  # clear screen, home cursor
    print(f"┏━━ TYPESHI TRAINING ━━ {path.name} ━━ {datetime.now():%H:%M:%S} ━━┓")

    if not alive:
        done = "train_runtime" in raw
        print("┃")
        print(f"┃  {'✅ RUN COMPLETE' if done else '💀 PROCESS NOT RUNNING'}")
    elif stale_s > 120:
        print(f"┃  ⚠️  log stale for {stale_s/60:.0f} min — possible hang")
    else:
        print("┃  🟢 running")

    if steps:
        pct, step, total, elapsed, eta, rate = steps[-1]
        filled = int(int(pct) / 100 * 30)
        print(f"┃  [{'█' * filled}{'░' * (30 - filled)}] {pct}%  "
              f"step {step}/{total}")
        print(f"┃  elapsed {elapsed}   eta {eta}   {rate}s/step")

    if losses:
        vals = [float(l) for l, _, _, _ in losses]
        _, ent, acc, epoch = losses[-1]
        print("┃")
        print(f"┃  loss  {vals[0]:.2f} → {vals[-1]:.3f}   {sparkline(vals)}")
        print(f"┃  epoch {float(epoch):.2f}   entropy {float(ent):.2f}   "
              f"token-acc {float(acc) * 100:.1f}%")
        recent = vals[-6:]
        print(f"┃  last 6: {'  '.join(f'{v:.3f}' for v in recent)}")

    for bad in ("Traceback", "out of memory", "Killed"):
        if bad in raw:
            print(f"┃  ❌ log contains: {bad!r}")

    print("┗" + "━" * 62)
    print(f"  refreshing every {REFRESH_S}s — ctrl-c to quit")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else newest_log()
    once = "--once" in sys.argv
    while True:
        render(path)
        if once:
            return
        time.sleep(REFRESH_S)


if __name__ == "__main__":
    main()
