"""V2-E0 Step 2 / A1 — firearm utility vs SNR on the BASELINE (Stage-A) encoder, with
the RECEIVER adapted through noise (freeze E=conv1..layer2, train R=layer3..fc a few
epochs on firearm CE with FEATURE-SPACE channel noise -- the same channel as A2 /
eval_privacy, NOT main.py's input-space ITIT noise). Reports recall at matched FPR<=2%
+ ROC-AUC over the 48k protocol (50 firearm via constants.firearm_file_paths + 48,800
background), averaged over 3 fresh-noise draws. sigma=0 no-adapt row validates the
evaluator (expect recall 50/50, AUC~1.0). -> v2e0_results/step2_a1.json

Run: .venv/bin/python v2e0_step2_a1.py
"""
import json, os, numpy as np, torch, torch.nn.functional as F
from types import SimpleNamespace
from pathlib import Path
from sklearn.metrics import roc_auc_score
import constants as Constants
from privacy.dataset import PrivacyTaskCoalitionDataset
from privacy.eval_privacy import build_datasets, load_backbone
from privacy._snr import snr_to_noise_std

SNRS = [0.0, -5.0, -10.0]; ADAPT_EPOCHS = 5; ADAPT_LR = 1e-4; EVAL_DRAWS = 3
RESDIR = "v2e0_results"; os.makedirs(RESDIR, exist_ok=True)
device = torch.device("cuda")
E_MODS = ["conv1", "bn1", "layer1", "layer2"]; R_MODS = ["layer3", "layer4", "fc"]

args = SimpleNamespace(data="data/GroupTestingDataset", task_num=2, background_K=0, GT_alg=1,
                       arch="resnet18", phase=False,
                       stage_b_ckpt="Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar",
                       batch_size=32, workers=8, seed=0)
torch.manual_seed(args.seed)

train_list, val_list = build_datasets(args)
all_wnids = sorted(set().union(*(ds.class_to_idx.keys() for ds in train_list + val_list)))
wmap = {w: i for i, w in enumerate(all_wnids)}

# 48k firearm-vs-background val (exact 50 firearm + all background)
fire = [(p, 1) for p, _ in val_list[0].samples if Path(p).name in Constants.firearm_file_paths]
bg = [(p, 0) for p, _ in val_list[1].samples]
assert len(fire) == 50, len(fire)
val_items = fire + bg
loader_fn, tf = val_list[1].loader, val_list[1].transform
print(f"A1 val: {len(fire)} firearm + {len(bg)} background")

class ValSet(torch.utils.data.Dataset):
    def __len__(self): return len(val_items)
    def __getitem__(self, i):
        p, y = val_items[i]
        return tf(loader_fn(p)).unsqueeze(0), y
valloader = torch.utils.data.DataLoader(ValSet(), batch_size=64, shuffle=False,
                                        num_workers=8, pin_memory=True)

def freeze_encoder(bb):
    for n, p in bb.named_parameters():
        p.requires_grad = not any(n.startswith(m + ".") or n == m for m in E_MODS)
    for m in E_MODS: getattr(bb, m).eval()
    for m in R_MODS: getattr(bb, m).train()

def eval_recall(bb, noise_std):
    bb.eval(); scores, labels = [], []
    with torch.no_grad():
        for imgs, y in valloader:
            imgs = imgs.to(device)
            out, _, _ = bb(imgs, noise_std, 0)
            scores.append(F.softmax(out, -1)[:, 1].cpu().numpy()); labels.append(y.numpy())
    s = np.concatenate(scores); y = np.concatenate(labels)
    thr = np.quantile(s[y == 0], 0.98)                 # FPR <= 2%
    return roc_auc_score(y, s), float((s[y == 1] > thr).mean())

# --- evaluator validation: sigma=0, no adaptation ---
bb, coded_pwr = load_backbone(args, device)
a0, r0 = eval_recall(bb, None)
print(f"[VALIDATE sigma=0 no-adapt] AUC={a0:.4f} recall@2%FPR={r0:.3f}  (expect ~1.0, 50/50)")

train_ds = PrivacyTaskCoalitionDataset(train_list, args, split="train", wnid_to_imagenet_idx=wmap)
results = [dict(snr=None, adapted=False, auc=a0, recall_fpr2=r0)]
for snr in SNRS:
    bb, coded_pwr = load_backbone(args, device)
    ns = snr_to_noise_std(snr, 1, coded_pwr, feat_channels=bb.layer3[0].conv1.in_channels)
    freeze_encoder(bb)
    opt = torch.optim.SGD([p for p in bb.parameters() if p.requires_grad],
                          lr=ADAPT_LR, momentum=0.9, weight_decay=1e-4)
    print(f"\n===== A1 SNR={snr} dB (std={ns:.4f}) — adapt R only ({ADAPT_EPOCHS} ep) =====", flush=True)
    for ep in range(ADAPT_EPOCHS):
        for m in R_MODS: getattr(bb, m).train()
        for m in E_MODS: getattr(bb, m).eval()
        dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.workers, pin_memory=True, drop_last=True)
        tot = 0.0; nb = 0
        for imgs, fire_t, _ in dl:
            imgs = imgs.to(device); fire_t = fire_t.to(device).reshape(-1)
            with torch.no_grad():
                pre = bb.encode(imgs)
            post, _, _ = bb.channel(pre, ns, 0)
            loss = F.cross_entropy(bb.decode(post), fire_t)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  ep{ep} firearm_CE={tot/max(nb,1):.4f}", flush=True)
    aucs, recs = [], []
    for d in range(EVAL_DRAWS):
        a, r = eval_recall(bb, ns); aucs.append(a); recs.append(r)
    row = dict(snr=snr, adapted=True, auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
               recall_fpr2=float(np.mean(recs)), recall_sd=float(np.std(recs)))
    results.append(row)
    print(f"SNR={snr} dB  recall@2%FPR={row['recall_fpr2']:.3f}+/-{row['recall_sd']:.3f}  AUC={row['auc']:.4f}", flush=True)

json.dump(results, open(os.path.join(RESDIR, "step2_a1.json"), "w"), indent=2)
print("\nV2E0_STEP2_A1_DONE")
for r in results:
    tag = "sigma0" if r["snr"] is None else f"{r['snr']:>5} dB"
    print(f"  {tag}: recall@2%FPR={r['recall_fpr2']:.3f}  AUC={r['auc']:.4f}")
