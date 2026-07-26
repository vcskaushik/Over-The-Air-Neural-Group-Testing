"""V2-E0 Step 1 — THE GATE: static noise-collapse probe (no training).

On the FROZEN Stage-A (baseline / lam_H=0) encoder, locate where each task's
decision collapses under channel noise, from cached clean pre-channel features:
  - firearm: receiver (decode) 2-way margin z_firearm - z_background  -> AUC & recall@FPR<=2%
  - class:   frozen sigma=0 worst-case adversary top-1 over background val -> leakage
Both are Monte-Carlo over fresh noise draws at each SNR (the "fraction-flipped vs
SNR" curve). sigma=0 row validates the path (firearm AUC~1.0 recall 50/50;
class top1 ~= the 27.8% baseline leakage).

GATE: as SNR drops the smaller-margin task collapses first (higher SNR). If class
collapses meaningfully ABOVE firearm -> a window may exist -> Step 2. Else coincide.

Run: .venv/bin/python v2e0_gate_probe.py
"""
import json, os, numpy as np, torch
from types import SimpleNamespace
from sklearn.metrics import roc_auc_score
import resnet_design2 as models
from privacy.adversary import AdversaryHead
from privacy.eval_privacy import build_datasets, load_backbone
from privacy._snr import snr_to_noise_std

OUT = "Trained_Models/StageC_OnStageA_ResNet18_advpretrained"
RESDIR = "v2e0_results"; os.makedirs(RESDIR, exist_ok=True)
N_BG = 4000            # background val sample for the probe (of 48,800)
DRAWS = 5             # fresh-noise draws per SNR
SNRS = [30, 25, 20, 17, 14, 11, 8, 5, 2, 0, -2, -5, -8, -11, -14, -17, -20, -25]
SEED = 0

args = SimpleNamespace(data="data/GroupTestingDataset", task_num=2, background_K=0,
                       GT_alg=1, arch="resnet18", phase=False, SNR=None,
                       stage_b_ckpt="Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar")
device = torch.device("cuda")
torch.manual_seed(SEED); g = torch.Generator(device="cuda").manual_seed(SEED)

# --- frozen backbone (firearm receiver) + frozen sigma=0 worst-case adversary ---
backbone, coded_pwr = load_backbone(args, device)
feat_ch = backbone.layer3[0].conv1.in_channels
advck = torch.load(os.path.join(OUT, "adversary.pth"), map_location="cpu", weights_only=False)
adversary = AdversaryHead(arch_name=args.arch, num_classes=advck["num_classes"], pretrained=False).to(device)
adversary.load_state_dict(advck["state_dict"]); adversary.eval()
for p in adversary.parameters(): p.requires_grad = False
print(f"coded_pwr={coded_pwr:.6f} feat_ch={feat_ch} adv_classes={advck['num_classes']}")

# --- datasets + unified wnid map (must match training) ---
train_list, val_list = build_datasets(args)
all_wnids = sorted(set().union(*(ds.class_to_idx.keys() for ds in train_list + val_list)))
wmap = {w: i for i, w in enumerate(all_wnids)}
from pathlib import Path

def cache_feats(samples, want_class):
    """Encode a list of image paths -> stash pre-channel feats (B,C,H,W) on GPU + class idx."""
    loader = val_list[1].loader; tf = val_list[1].transform
    feats, cls = [], []
    bs = 64; buf = []
    def flush(buf):
        x = torch.stack(buf).unsqueeze(1).to(device)     # (b,1,C,H,W)
        with torch.no_grad():
            pre = backbone.encode(x)[:, 0]               # (b, C2,H2,W2) pre-channel, K=1
        feats.append(pre)
    for path, wnid in samples:
        buf.append(tf(loader(path)))
        if want_class: cls.append(wmap[wnid])
        if len(buf) == bs: flush(buf); buf = []
    if buf: flush(buf)
    F = torch.cat(feats, 0)
    return F, (torch.tensor(cls, device=device) if want_class else None)

# firearm val = task0 (all); background val = sample of task1
fire_samples = [(p, Path(p).parent.name) for p, _ in val_list[0].samples]
bg_all = [(p, Path(p).parent.name) for p, _ in val_list[1].samples]
rng = np.random.default_rng(SEED)
bg_idx = rng.choice(len(bg_all), size=min(N_BG, len(bg_all)), replace=False)
bg_samples = [bg_all[i] for i in bg_idx]
print(f"firearm val={len(fire_samples)}  background sample={len(bg_samples)} (of {len(bg_all)})")

Ffire, _ = cache_feats(fire_samples, want_class=False)
Fbg, Cbg = cache_feats(bg_samples, want_class=True)
print(f"cached feats: fire={tuple(Ffire.shape)} bg={tuple(Fbg.shape)}")

def add_noise(F, std):
    if std is None: return F
    return F + torch.normal(0.0, float(std), size=F.shape, generator=g, device=device)

def _margins(F, std):
    out = []
    with torch.no_grad():
        for i in range(0, F.shape[0], 1024):
            z = backbone.decode(add_noise(F[i:i+1024], std))
            out.append((z[:, 1] - z[:, 0]).cpu().numpy())
    return np.concatenate(out)

def firearm_metrics(std):
    mf = _margins(Ffire, std); mb = _margins(Fbg, std)
    y = np.r_[np.ones_like(mf), np.zeros_like(mb)]; s = np.r_[mf, mb]
    auc = roc_auc_score(y, s)
    thr = np.quantile(mb, 0.98)                       # FPR <= 2% on background
    recall = float((mf > thr).mean())
    return auc, recall

def class_top1(std):
    correct = tot = 0
    with torch.no_grad():
        for i in range(0, Fbg.shape[0], 512):
            fb = add_noise(Fbg[i:i+512], std)
            pred = adversary(fb).argmax(1)
            correct += (pred == Cbg[i:i+512]).sum().item(); tot += fb.shape[0]
    return correct / tot

rows = []
for snr in [None] + SNRS:
    std = snr_to_noise_std(snr, args.GT_alg, coded_pwr, feat_channels=feat_ch)
    aucs, recs, accs = [], [], []
    reps = 1 if snr is None else DRAWS
    for _ in range(reps):
        a, r = firearm_metrics(std); aucs.append(a); recs.append(r)
        accs.append(class_top1(std))
    row = dict(snr=snr, noise_std=std,
               fire_auc=float(np.mean(aucs)), fire_auc_sd=float(np.std(aucs)),
               fire_recall_fpr2=float(np.mean(recs)), fire_recall_sd=float(np.std(recs)),
               class_top1=float(np.mean(accs)), class_top1_sd=float(np.std(accs)))
    rows.append(row)
    tag = "sigma0" if snr is None else f"{snr:>4} dB"
    print(f"[{tag}] std={0.0 if std is None else std:7.4f}  fireAUC={row['fire_auc']:.4f}"
          f"  recall@2%FPR={row['fire_recall_fpr2']:.3f}  classTop1={row['class_top1']*100:6.2f}%")

with open(os.path.join(RESDIR, "gate.json"), "w") as f:
    json.dump({"coded_pwr": coded_pwr, "n_bg": len(bg_samples), "draws": DRAWS, "rows": rows}, f, indent=2)
print("V2E0_GATE_DONE -> v2e0_results/gate.json")
