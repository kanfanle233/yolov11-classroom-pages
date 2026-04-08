import argparse
import subprocess
import sys
from pathlib import Path


LEGACY_NOTICE = (
    "[LEGACY] scripts/09b_run_pipeline.py is deprecated. "
    "Use scripts/09_run_pipeline.py (formal fixed-schema pipeline)."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy wrapper for old entrypoint.")
    parser.add_argument("--allow_legacy", type=int, default=0, help="set 1 to execute forwarded run")
    args, extra = parser.parse_known_args()
    print(LEGACY_NOTICE)
    if int(args.allow_legacy) != 1:
        print("[STOP] Legacy wrapper exited. Re-run with --allow_legacy 1 if you must.")
        return

    root = Path(__file__).resolve().parents[1]
    target = root / "scripts" / "09_run_pipeline.py"
    cmd = [sys.executable, str(target)] + extra
    subprocess.run(cmd, check=False, cwd=str(root))


if __name__ == "__main__":
    main()
