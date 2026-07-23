import torch
import resnet_design2 as models
from privacy.train_invariance import save_invariance_ckpt


def test_save_invariance_ckpt_is_main_resume_compatible(tmp_path):
    backbone = models.resnet18(pretrained=False, gt=True, phase=False)
    path = tmp_path / "invariance_final.pth.tar"
    save_invariance_ckpt(backbone, coded_pwr=2.5, epoch=7, path=str(path))
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    assert "state_dict" in ckpt and "coded_pwr" in ckpt and "epoch" in ckpt
    assert ckpt["coded_pwr"] == 2.5 and ckpt["epoch"] == 7
    # main.py --resume expects the module. prefix.
    assert all(k.startswith("module.") for k in ckpt["state_dict"])
