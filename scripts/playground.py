"""Local playground: watch the motor model type, live, on a fake keyboard.

    uv run python scripts/playground.py                        # newest checkpoint
    uv run python scripts/playground.py --checkpoint checkpoints/motor-phase2

Then open http://localhost:8765.

The port binds BEFORE the model loads, so the page is up and reporting
progress while weights stream in -- a cold start on a new base model pulls
9.3 GB and the old build-then-bind order left the browser staring at a dead
localhost for minutes.

Serving logic lives in typeshi.portal; this file is only the CLI. Stdlib HTTP
only -- no new dependencies for a toy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from typeshi.portal.registry import discover
from typeshi.portal.server import Portal, serve

HERE = Path(__file__).parent
PREFERRED = Path("checkpoints/motor-phase2")


def default_checkpoint(root: Path) -> Path | None:
    """The phase-2 model if it is there, else the newest loadable save."""
    if (PREFERRED / "adapter_config.json").exists():
        return PREFERRED
    found = discover(root)
    return found[0].path if found else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="defaults to checkpoints/motor-phase2, else the "
                         "newest save under --checkpoint-root")
    ap.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--no-load", action="store_true",
                    help="start the server without loading a model; pick one "
                         "in the UI")
    args = ap.parse_args()

    portal = Portal(HERE / "playground.html", args.checkpoint_root,
                    args.max_chars)
    server = serve(portal, args.port)
    print(f"portal ready:  http://localhost:{args.port}   (ctrl-c to stop)")

    if not args.no_load:
        ckpt = args.checkpoint or default_checkpoint(args.checkpoint_root)
        if ckpt is None:
            print("no checkpoint found; pick one in the UI "
                  "(or pass --checkpoint)")
        else:
            print(f"loading {ckpt} in the background ...")
            portal.registry.load_async(ckpt)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")


if __name__ == "__main__":
    main()
