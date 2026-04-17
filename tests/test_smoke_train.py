"""End-to-end smoke test for privacy.train_privacy.

Marked slow + gpu — runs only if the dataset and a CUDA device are present.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "GroupTestingDataset"
STAGE_A_CKPT = REPO_ROOT / "Trained_Models" / "SmokeTest" / "checkpoint.pth.tar"


@pytest.mark.slow
@pytest.mark.gpu
def test_train_privacy_smoke(tmp_path):
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

    out_dir = tmp_path / "PrivacySmoke"
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"), "-u", "-m", "privacy.train_privacy",
        "--stage-a-ckpt", str(STAGE_A_CKPT),
        "--data", str(DATA_ROOT), "--task-num", "2", "--background-K", "0",
        "--GT-alg", "1", "-a", "resnet18", "--priv-loss", "entropy",
        "--lambda", "1.0", "--k-adv", "2",
        "--stage-b-epochs", "1", "--recovery-epochs", "0",
        "--batch-size", "8", "-j", "2", "-valj", "1", "--print-freq", "1",
        "--output_dir", str(out_dir), "--log-name", "smoke.log",
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"train_privacy failed:\n{proc.stdout}\n{proc.stderr}"
    assert (out_dir / "stage_b_final.pth.tar").exists()
