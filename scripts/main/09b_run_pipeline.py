import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatibility alias for scripts/main/09_run_pipeline.py.")
    _, extra = parser.parse_known_args()

    root = Path(__file__).resolve().parents[2]
    target = root / "scripts" / "main" / "09_run_pipeline.py"
    cmd = [sys.executable, str(target)] + extra
    completed = subprocess.run(cmd, check=False, cwd=str(root))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
