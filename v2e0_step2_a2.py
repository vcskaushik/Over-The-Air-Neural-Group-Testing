"""V2-E0 Step 2 / A2 — worst-case class leakage vs SNR on the BASELINE (Stage-A) encoder,
with the adversary RETRAINED THROUGH NOISE at each SNR (the honest attacker; the decisive
test of whether a noise-adapted adversary recovers the class signal the static gate lost).

For each SNR: train a pretrained-init adversary 30 epochs with channel noise at that SNR,
then evaluate_leakage over the full 48,800 background val, averaged over 3 fresh-noise draws
(mean +/- sd). Saves per-SNR leakage to v2e0_results/step2_a2.json.

Run: .venv/bin/python v2e0_step2_a2.py
"""
import json, os, numpy as np, torch
from types import SimpleNamespace
import resnet_design2 as models
from privacy.adversary import AdversaryHead
from privacy.dataset import PrivacyTaskCoalitionDataset
from privacy.eval_privacy import (build_datasets, load_backbone, train_fresh_adversary,
                                  evaluate_leakage, build_background_val_index, BackgroundValDataset)

SNRS = [0.0, -5.0, -10.0]
EVAL_DRAWS = 3
RESDIR = "v2e0_results"; os.makedirs(RESDIR, exist_ok=True)
# NOTE: explicit index (cuda:0) is REQUIRED — eval_privacy passes gpu=device.index to
# backbone.channel to move the AWGN to GPU; torch.device("cuda").index is None, which
# leaves the noise on CPU and crashes at sigma>0 (latent bug, never hit at sigma=0).
device = torch.device("cuda:0")

base = dict(data="data/GroupTestingDataset", task_num=2, background_K=0, GT_alg=1,
            arch="resnet18", phase=False,
            stage_b_ckpt="Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar",
            adv_init="pretrained", stage_c_epochs=30, adv_lr=1e-3, momentum=0.9,
            weight_decay=1e-4, batch_size=32, workers=8, val_workers=4, print_freq=200, seed=0)

results = []
for snr in SNRS:
    outdir = f"Trained_Models/StageC_OnStageA_ResNet18_advpretrained_snr{snr:g}"
    os.makedirs(outdir, exist_ok=True)
    args = SimpleNamespace(**base, SNR=snr, output_dir=outdir, log_name="eval_c.log")
    torch.manual_seed(args.seed)
    log = open(os.path.join(outdir, args.log_name), "w")
    backbone, coded_pwr = load_backbone(args, device)
    train_list, val_list = build_datasets(args)
    all_wnids = sorted(set().union(*(ds.class_to_idx.keys() for ds in train_list + val_list)))
    wmap = {w: i for i, w in enumerate(all_wnids)}
    train_ds = PrivacyTaskCoalitionDataset(train_list, args, split="train", wnid_to_imagenet_idx=wmap)
    adversary = AdversaryHead(arch_name=args.arch, num_classes=train_ds.num_imagenet_classes,
                              pretrained=True).to(device)
    print(f"\n===== A2 SNR={snr} dB (coded_pwr={coded_pwr:.4f}) — train adversary through noise =====", flush=True)
    train_fresh_adversary(backbone, adversary, train_ds, args, device, log, coded_pwr=coded_pwr)
    bg_index = build_background_val_index(val_list, wmap)
    leak_val = BackgroundValDataset(bg_index, val_list[1].loader, val_list[1].transform)
    draws = []
    for d in range(EVAL_DRAWS):
        m = evaluate_leakage(backbone, adversary, leak_val, args, device, coded_pwr=coded_pwr)
        draws.append(m["top1_imagenet_acc"]); print(f"  draw {d}: {m['top1_imagenet_acc']*100:.2f}%", flush=True)
    row = dict(snr=snr, leak_mean=float(np.mean(draws)), leak_sd=float(np.std(draws)), draws=draws)
    results.append(row); log.write("a2: " + json.dumps(row) + "\n"); log.close()
    torch.save({"state_dict": adversary.state_dict(), "num_classes": train_ds.num_imagenet_classes},
               os.path.join(outdir, "adversary.pth"))
    print(f"SNR={snr} dB  worst-case leakage = {row['leak_mean']*100:.2f}% +/- {row['leak_sd']*100:.2f}", flush=True)

json.dump(results, open(os.path.join(RESDIR, "step2_a2.json"), "w"), indent=2)
print("\nV2E0_STEP2_A2_DONE"); [print(f"  {r['snr']:>5} dB : {r['leak_mean']*100:5.2f}% +/- {r['leak_sd']*100:.2f}") for r in results]
