import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def test_distributed_tp_correctness_under_torchrun():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is required to run distributed correctness tests.")

    root = Path(__file__).resolve().parents[1]
    script = root / "tests" / "distributed_tp_correctness.py"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(script),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "distributed TP correctness tests passed" in result.stdout
