#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path


TRITON_COMMIT = "5f3f125e8f63c24613f1f73b937442864f263f94"
AITER_COMMIT = "38e534a1c07278b85989dce5a5a6fc18f98742b0"

TRITON = (
    "triton @ "
    f"git+https://github.com/triton-lang/triton.git@{TRITON_COMMIT}"
)
TRITON_KERNELS = (
    "triton_kernels @ "
    f"git+https://github.com/triton-lang/triton.git@{TRITON_COMMIT}"
    "#subdirectory=python/triton_kernels"
)
AITER = (
    "amd-aiter @ "
    f"git+https://github.com/ROCm/aiter.git@{AITER_COMMIT}"
)


def install(requirement, *options, env=None):
    subprocess.run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--force-reinstall",
        "--no-deps",
        *options,
        requirement,
    ], check=True, env=env)


def main():
    install(TRITON)
    install(TRITON_KERNELS)

    aiter_env = os.environ.copy()
    aiter_env["AITER_USE_SYSTEM_TRITON"] = "1"
    aiter_env["AITER_TRITON_ONLY"] = "1"
    install(
        AITER,
        "--no-build-isolation",
        "--no-cache-dir",
        env=aiter_env,
    )

    subprocess.run([
        sys.executable,
        str(Path(__file__).parent / "gptoss" / "gptoss-aot.py"),
        "--help",
    ], check=True)


if __name__ == "__main__":
    main()
