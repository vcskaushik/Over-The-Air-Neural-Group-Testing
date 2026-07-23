"""End-to-end smoke test for privacy.train_invariance. Slow + gpu."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "GroupTestingDataset"
STAGE_A_CKPT = REPO_ROOT / "Trained_Models" / "SmokeTest" / "checkpoint.pth.tar"


@pytest.mark.slow
@pytest.mark.gpu
def test_train_invariance_smoke(tmp_path):
    if not DATA_ROOT.exists():
        pytest.skip(f"Missing dataset at {DATA_ROOT}")
    if not STAGE_A_CKPT.exists():
        pytest.skip(f"Missing Stage A checkpoint at {STAGE_A_CKPT}")
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("torch not installed")

    out_dir = tmp_path / "InvarianceSmoke"
    cmd = [
        sys.executable, "-u", "-m", "privacy.train_invariance",
        "--stage-a-ckpt", str(STAGE_A_CKPT),
        "--data", str(DATA_ROOT), "--task-num", "2", "--background-K", "0",
        "--GT-alg", "1", "-a", "resnet18",
        "--hsic-lambda", "1.0", "--hsic-extractor", "random", "--hsic-num-random", "1",
        "--pk-classes", "4", "--pk-per-class", "4", "--pk-firearm", "4",
        "--epochs", "1", "--batch-size", "32",
        "-j", "2", "-valj", "1", "--print-freq", "1",
        "--output_dir", str(out_dir), "--log-name", "smoke.log",
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, f"train_invariance failed:\n{proc.stdout}\n{proc.stderr}"
    assert (out_dir / "invariance_final.pth.tar").exists()
