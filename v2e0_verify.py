"""V2-E0 VERIFICATION (HANDOFF-v2-E0-verify.md) — close the null at the letter level.
Reuses the E0 A1/A2 logic on frozen checkpoints. Writes v2e0_results/verify.json.

V1: baseline @ {-7,-8} dB — A1 (receiver-adapted, 10 ep) recall@2%FPR + A2 leakage. (window band)
V2: lam20/50/70 @ sigma=0 — matched-FPR recall (no adapt). (fix the argmax mismatch)
V3: combo lam30 + -5 dB, SEED 1 — A1+A2 vs seed-0's 9.66%. (E3 complementarity)
V4: baseline @ {-5,-10} dB — A2 60-epoch Stage-C. (convergence check)
"""
import json, os, numpy as np, torch, torch.nn.functional as F
from types import SimpleNamespace
from pathlib import Path
from sklearn.metrics import roc_auc_score
import constants as Constants
from privacy.adversary import AdversaryHead
from privacy.dataset import PrivacyTaskCoalitionDataset
from privacy.eval_privacy import (build_datasets, load_backbone, train_fresh_adversary,
                                  evaluate_leakage, build_background_val_index, BackgroundValDataset)
from privacy._snr import snr_to_noise_std

device = torch.device("cuda:0")
RESDIR = "v2e0_results"; os.makedirs(RESDIR, exist_ok=True)
E_MODS = ["conv1", "bn1", "layer1", "layer2"]; R_MODS = ["layer3", "layer4", "fc"]
SA = "Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar"
CK = lambda l: f"Trained_Models/Invariance_ResNet18_lam{l}/invariance_final.pth.tar"
base = dict(data="data/GroupTestingDataset", task_num=2, background_K=0, GT_alg=1,
            arch="resnet18", phase=False, batch_size=32, workers=8, val_workers=4,
            adv_init="pretrained", adv_lr=1e-3, momentum=0.9, weight_decay=1e-4, print_freq=400)

torch.manual_seed(0)
args0 = SimpleNamespace(**base, SNR=None, stage_b_ckpt=SA, stage_c_epochs=30, seed=0)
train_list, val_list = build_datasets(args0)
all_wnids = sorted(set().union(*(ds.class_to_idx.keys() for ds in train_list + val_list)))
wmap = {w: i for i, w in enumerate(all_wnids)}
fire = [(p, 1) for p, _ in val_list[0].samples if Path(p).name in Constants.firearm_file_paths]
bg = [(p, 0) for p, _ in val_list[1].samples]; assert len(fire) == 50
val_items = fire + bg; loader_fn, tf = val_list[1].loader, val_list[1].transform
class ValSet(torch.utils.data.Dataset):
    def __len__(self): return len(val_items)
    def __getitem__(self, i):
        p, y = val_items[i]; return tf(loader_fn(p)).unsqueeze(0), y
valloader = torch.utils.data.DataLoader(ValSet(), batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
train_ds = PrivacyTaskCoalitionDataset(train_list, args0, split="train", wnid_to_imagenet_idx=wmap)
bg_index = build_background_val_index(val_list, wmap)
leak_val = BackgroundValDataset(bg_index, val_list[1].loader, val_list[1].transform)

def eval_recall(bb, ns):
    bb.eval(); s, y = [], []
    with torch.no_grad():
        for imgs, lab in valloader:
            out, _, _ = bb(imgs.to(device), ns, 0)
            s.append(F.softmax(out, -1)[:, 1].cpu().numpy()); y.append(lab.numpy())
    s = np.concatenate(s); y = np.concatenate(y); thr = np.quantile(s[y == 0], 0.98)
    return float(roc_auc_score(y, s)), float((s[y == 1] > thr).mean())

def a1_run(ckpt, snr, adapt_epochs=10, seed=0):
    torch.manual_seed(seed)
    bb, cp = load_backbone(SimpleNamespace(**base, SNR=snr, stage_b_ckpt=ckpt, stage_c_epochs=30, seed=seed), device)
    ns = snr_to_noise_std(snr, 1, cp, feat_channels=bb.layer3[0].conv1.in_channels)
    if snr is None:
        a, r = eval_recall(bb, None); return dict(auc=a, recall_fpr2=r, adapted=False)
    for n, p in bb.named_parameters():
        p.requires_grad = not any(n.startswith(m + ".") or n == m for m in E_MODS)
    opt = torch.optim.SGD([p for p in bb.parameters() if p.requires_grad], lr=1e-4, momentum=0.9, weight_decay=1e-4)
    for ep in range(adapt_epochs):
        for m in R_MODS: getattr(bb, m).train()
        for m in E_MODS: getattr(bb, m).eval()
        dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        for imgs, ft, _ in dl:
            with torch.no_grad(): pre = bb.encode(imgs.to(device))
            post, _, _ = bb.channel(pre, ns, 0)
            loss = F.cross_entropy(bb.decode(post), ft.to(device).reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    aucs, recs = zip(*[eval_recall(bb, ns) for _ in range(3)])
    return dict(auc=float(np.mean(aucs)), recall_fpr2=float(np.mean(recs)), recall_sd=float(np.std(recs)),
                adapted=True, adapt_epochs=adapt_epochs)

def a2_run(ckpt, snr, epochs=30, seed=0):
    torch.manual_seed(seed)
    args = SimpleNamespace(**base, SNR=snr, stage_b_ckpt=ckpt, stage_c_epochs=epochs,
                           output_dir=f"{RESDIR}/tmp", log_name="l.log", seed=seed)
    os.makedirs(args.output_dir, exist_ok=True)
    bb, cp = load_backbone(args, device)
    adv = AdversaryHead(arch_name="resnet18", num_classes=train_ds.num_imagenet_classes, pretrained=True).to(device)
    log = open(os.path.join(args.output_dir, "l.log"), "w")
    train_fresh_adversary(bb, adv, train_ds, args, device, log, coded_pwr=cp); log.close()
    draws = [evaluate_leakage(bb, adv, leak_val, args, device, coded_pwr=cp)["top1_imagenet_acc"] for _ in range(3)]
    return dict(leak_mean=float(np.mean(draws)), leak_sd=float(np.std(draws)), epochs=epochs, seed=seed)

out = {}
def save(): json.dump(out, open(f"{RESDIR}/verify.json", "w"), indent=2)

# V1 — window band
out["V1"] = []
for snr in [-7.0, -8.0]:
    print(f"\n### V1 baseline SNR={snr} dB ###", flush=True)
    r = dict(snr=snr, a1=a1_run(SA, snr, adapt_epochs=10), a2=a2_run(SA, snr, epochs=30))
    print("  ", r["a1"], r["a2"], flush=True); out["V1"].append(r); save()

# V2 — matched-FPR recall for HSIC lam20/50/70 (no adaptation)
out["V2"] = []
for l in [20, 50, 70]:
    r = dict(lam=l, a1=a1_run(CK(l), None))
    print(f"### V2 lam{l} sigma0 matched-FPR recall: {r['a1']}", flush=True); out["V2"].append(r); save()

# V3 — combo lam30 + -5 dB, second seed
print("\n### V3 combo lam30 + -5 dB seed=1 ###", flush=True)
out["V3"] = dict(seed1_a1=a1_run(CK(30), -5.0, adapt_epochs=5, seed=1),
                 seed1_a2=a2_run(CK(30), -5.0, epochs=30, seed=1),
                 seed0_leak_ref=0.0966)
print("  ", out["V3"], flush=True); save()

# V4 — 60-epoch convergence at -5,-10 dB baseline
out["V4"] = []
for snr in [-5.0, -10.0]:
    print(f"\n### V4 baseline SNR={snr} dB 60-ep Stage-C ###", flush=True)
    r = dict(snr=snr, a2_60ep=a2_run(SA, snr, epochs=60))
    print("  ", r, flush=True); out["V4"].append(r); save()

print("\nV2E0_VERIFY_DONE")
