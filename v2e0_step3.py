"""V2-E0 Step 3 — characterize the confirmed window. For selected (checkpoint, SNR):
  A1 = firearm recall@FPR<=2% with R adapted through noise (feature-space channel),
  A2 = worst-case class leakage with the adversary retrained through noise.
Answers: window lower edge (baseline @ -15 dB) and whether noise COMPLEMENTS HSIC
(lam30/lam100 @ -5/-10 dB vs their sigma=0 leakage). -> v2e0_results/step3.json
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
L30 = "Trained_Models/Invariance_ResNet18_lam30/invariance_final.pth.tar"
L100 = "Trained_Models/Invariance_ResNet18_lam100/invariance_final.pth.tar"

# (tag, ckpt, SNR, do_a1, do_a2)
COMBOS = [
    ("baseline", SA,  -15.0, True,  True),
    ("lam30",    L30, None,  True,  False),
    ("lam30",    L30, -5.0,  True,  True),
    ("lam30",    L30, -10.0, True,  True),
    ("lam100",   L100, None, True,  False),
    ("lam100",   L100, -5.0, True,  True),
]
base = dict(data="data/GroupTestingDataset", task_num=2, background_K=0, GT_alg=1,
            arch="resnet18", phase=False, batch_size=32, workers=8, val_workers=4,
            adv_init="pretrained", stage_c_epochs=30, adv_lr=1e-3, momentum=0.9,
            weight_decay=1e-4, print_freq=400, seed=0)
torch.manual_seed(0)

# shared datasets / maps / 48k firearm val
args0 = SimpleNamespace(**base, SNR=None, stage_b_ckpt=SA)
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
    return roc_auc_score(y, s), float((s[y == 1] > thr).mean())

def a1_run(ckpt, snr):
    bb, cp = load_backbone(SimpleNamespace(**base, SNR=snr, stage_b_ckpt=ckpt), device)
    ns = snr_to_noise_std(snr, 1, cp, feat_channels=bb.layer3[0].conv1.in_channels)
    if snr is None:
        a, r = eval_recall(bb, None); return dict(auc=a, recall_fpr2=r, adapted=False)
    for n, p in bb.named_parameters():
        p.requires_grad = not any(n.startswith(m + ".") or n == m for m in E_MODS)
    opt = torch.optim.SGD([p for p in bb.parameters() if p.requires_grad], lr=1e-4, momentum=0.9, weight_decay=1e-4)
    for ep in range(5):
        for m in R_MODS: getattr(bb, m).train()
        for m in E_MODS: getattr(bb, m).eval()
        dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        for imgs, ft, _ in dl:
            with torch.no_grad(): pre = bb.encode(imgs.to(device))
            post, _, _ = bb.channel(pre, ns, 0)
            loss = F.cross_entropy(bb.decode(post), ft.to(device).reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    aucs, recs = zip(*[eval_recall(bb, ns) for _ in range(3)])
    return dict(auc=float(np.mean(aucs)), recall_fpr2=float(np.mean(recs)), recall_sd=float(np.std(recs)), adapted=True)

def a2_run(ckpt, snr):
    args = SimpleNamespace(**base, SNR=snr, stage_b_ckpt=ckpt, output_dir=f"{RESDIR}/tmp", log_name="l.log")
    os.makedirs(args.output_dir, exist_ok=True)
    bb, cp = load_backbone(args, device)
    adv = AdversaryHead(arch_name="resnet18", num_classes=train_ds.num_imagenet_classes, pretrained=True).to(device)
    log = open(os.path.join(args.output_dir, "l.log"), "w")
    train_fresh_adversary(bb, adv, train_ds, args, device, log, coded_pwr=cp); log.close()
    draws = [evaluate_leakage(bb, adv, leak_val, args, device, coded_pwr=cp)["top1_imagenet_acc"] for _ in range(3)]
    return dict(leak_mean=float(np.mean(draws)), leak_sd=float(np.std(draws)))

results = []
for tag, ckpt, snr, do_a1, do_a2 in COMBOS:
    print(f"\n##### {tag} SNR={snr} (a1={do_a1} a2={do_a2}) #####", flush=True)
    row = dict(tag=tag, snr=snr)
    if do_a1: row["a1"] = a1_run(ckpt, snr); print("  A1", row["a1"], flush=True)
    if do_a2: row["a2"] = a2_run(ckpt, snr); print("  A2", row["a2"], flush=True)
    results.append(row); json.dump(results, open(f"{RESDIR}/step3.json", "w"), indent=2)
print("\nV2E0_STEP3_DONE")
