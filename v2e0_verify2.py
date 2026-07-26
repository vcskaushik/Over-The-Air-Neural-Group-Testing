"""V2-E0 verify (round 2) — clinch the 60-epoch matched comparison after V4 showed the
through-noise adversary under-converges at 30 ep. Two points, both worst-case Stage-C @ 60 ep:
  - HSIC lam30 @ sigma=0  (does clean-adversary HSIC stay ~18.6%? = converged at 30 ep)
  - baseline  @ -7 dB     (does the 30-ep window candidate 16.73% close at 60 ep?)
-> v2e0_results/verify2.json
"""
import json, os, numpy as np, torch
from types import SimpleNamespace
from privacy.adversary import AdversaryHead
from privacy.dataset import PrivacyTaskCoalitionDataset
from privacy.eval_privacy import (build_datasets, load_backbone, train_fresh_adversary,
                                  evaluate_leakage, build_background_val_index, BackgroundValDataset)

device = torch.device("cuda:0"); RESDIR = "v2e0_results"
SA = "Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar"
L30 = "Trained_Models/Invariance_ResNet18_lam30/invariance_final.pth.tar"
base = dict(data="data/GroupTestingDataset", task_num=2, background_K=0, GT_alg=1, arch="resnet18",
            phase=False, batch_size=32, workers=8, val_workers=4, adv_init="pretrained",
            adv_lr=1e-3, momentum=0.9, weight_decay=1e-4, print_freq=400)
torch.manual_seed(0)
a0 = SimpleNamespace(**base, SNR=None, stage_b_ckpt=SA, stage_c_epochs=60, seed=0)
train_list, val_list = build_datasets(a0)
wmap = {w: i for i, w in enumerate(sorted(set().union(*(ds.class_to_idx.keys() for ds in train_list + val_list))))}
train_ds = PrivacyTaskCoalitionDataset(train_list, a0, split="train", wnid_to_imagenet_idx=wmap)
leak_val = BackgroundValDataset(build_background_val_index(val_list, wmap), val_list[1].loader, val_list[1].transform)

def a2(ckpt, snr):
    torch.manual_seed(0)
    args = SimpleNamespace(**base, SNR=snr, stage_b_ckpt=ckpt, stage_c_epochs=60,
                           output_dir=f"{RESDIR}/tmp", log_name="l.log", seed=0)
    bb, cp = load_backbone(args, device)
    adv = AdversaryHead(arch_name="resnet18", num_classes=train_ds.num_imagenet_classes, pretrained=True).to(device)
    log = open(f"{RESDIR}/tmp_l.log", "w"); train_fresh_adversary(bb, adv, train_ds, args, device, log, coded_pwr=cp); log.close()
    draws = [evaluate_leakage(bb, adv, leak_val, args, device, coded_pwr=cp)["top1_imagenet_acc"] for _ in range(3)]
    return dict(leak_mean=float(np.mean(draws)), leak_sd=float(np.std(draws)))

out = {}
for tag, ck, snr in [("HSIC_lam30_sigma0_60ep", L30, None), ("baseline_-7dB_60ep", SA, -7.0)]:
    print(f"\n### {tag} ###", flush=True); out[tag] = a2(ck, snr)
    print("  ", out[tag], flush=True); json.dump(out, open(f"{RESDIR}/verify2.json", "w"), indent=2)
print("\nV2E0_VERIFY2_DONE")
