"""End-to-end smoke test for privacy.eval_privacy.

Requires a Stage-B checkpoint (produced by privacy.train_privacy).
Marked slow + gpu.
"""
import os
import sys
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "GroupTestingDataset"


@pytest.mark.slow
@pytest.mark.gpu
def test_eval_privacy_smoke(tmp_path):
    if not DATA_ROOT.exists():
        pytest.skip(f"Missing dataset at {DATA_ROOT}")
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("torch not installed")

    # Find any Stage-B checkpoint we can use.
    ckpts = list(REPO_ROOT.glob("Trained_Models/**/stage_b_final.pth.tar"))
    if not ckpts:
        pytest.skip("No stage_b_final.pth.tar checkpoint in Trained_Models/; run privacy.train_privacy first")
    ckpt = ckpts[0]

    out_dir = tmp_path / "EvalC"
    cmd = [
        sys.executable, "-u", "-m", "privacy.eval_privacy",
        "--stage-b-ckpt", str(ckpt),
        "--data", str(DATA_ROOT), "--task-num", "2", "--background-K", "0",
        "--GT-alg", "1", "-a", "resnet18",
        "--stage-c-epochs", "1", "--batch-size", "8",
        "-j", "2", "-valj", "1", "--print-freq", "5",
        "--output_dir", str(out_dir),
    ]
    env = os.environ.copy(); env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"eval_privacy failed:\n{proc.stdout}\n{proc.stderr}"
    leakage_path = out_dir / "leakage.json"
    assert leakage_path.exists()
    import json
    with open(leakage_path) as f:
        m = json.load(f)
    assert "top1_imagenet_acc" in m
