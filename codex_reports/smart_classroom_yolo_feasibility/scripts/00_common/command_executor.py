from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def ps_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def to_powershell_command(argv: List[str], env: Dict[str, str] | None = None) -> str:
    lines: List[str] = []
    if env:
        for key, value in env.items():
            lines.append(f"$env:{key}={ps_quote(value)}")
    cmd = "& " + " ".join(ps_quote(arg) for arg in argv)
    lines.append(cmd)
    return "\n".join(lines)


def run_command(argv: List[str], *, cwd: Path, log_file: Path, env: Dict[str, str] | None = None) -> Tuple[int, str]:
    run_env = os.environ.copy()
    if env:
        run_env.update({k: str(v) for k, v in env.items()})

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
            env=run_env,
        )
    return int(proc.returncode), str(log_file)
