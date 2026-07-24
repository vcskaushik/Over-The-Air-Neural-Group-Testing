"""Stage C: train a fresh-init adversary from scratch on a frozen privacy-fine-tuned encoder.

Reports its top-1 ImageNet accuracy (ITIT) or per-class mean ROC-AUC + mAP (GTGT-FM)
as the honest leakage measurement.

Usage:
    python -m privacy.eval_privacy \
        --stage-b-ckpt Trained_Models/PrivacySmoke/stage_b_final.pth.tar \
        --data data/GroupTestingDataset --task-num 2 --background-K 0 \
        --GT-alg 1 -a resnet18 --stage-c-epochs 5 --batch-size 16 \
        --output_dir Trained_Models/PrivacySmoke/EvalC
"""
import argparse
import json
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import resnet_design2 as models
from privacy.adversary import AdversaryHead
from privacy.dataset import PrivacyTaskCoalitionDataset
from privacy._snr import snr_to_noise_std


def get_parser():
    p = argparse.ArgumentParser("Privacy-preserving OTA-NGT — Stage C honest leakage eval")
    p.add_argument("--data", required=True)
    p.add_argument("--task-num", type=int, default=2)
    p.add_argument("--background-K", type=int, required=True)
    p.add_argument("--GT-alg", type=int, choices=[1, 2], required=True)
    p.add_argument("-a", "--arch", required=True)
    p.add_argument("--phase", action="store_true")
    p.add_argument("--SNR", type=float, default=None)
    p.add_argument("--stage-b-ckpt", required=True)
    p.add_argument("--adv-init", choices=["kaiming", "pretrained"], default="kaiming")
    p.add_argument("--stage-c-epochs", type=int, default=60)
    p.add_argument("--adv-lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("-j", "--workers", type=int, default=8)
    p.add_argument("-valj", "--val-workers", type=int, default=4)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--log-name", default="eval_c.log")
    p.add_argument("--print-freq", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    return p


def build_datasets(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_list, val_list = [], []
    for folder_idx in range(args.task_num):
        train_list.append(datasets.ImageFolder(os.path.join(args.data, str(folder_idx), "train"), train_t))
        val_list.append(datasets.ImageFolder(os.path.join(args.data, str(folder_idx), "val"), val_t))
    return train_list, val_list


def build_background_val_index(val_list, wnid_to_imagenet_idx):
    """Flat (path, imagenet_idx) over ALL background (task>=1) val images.

    The ~300-sample PrivacyTaskCoalitionDataset val undercounts leakage (M9);
    Stage-C leakage is measured over the full background val set instead.
    """
    from pathlib import Path
    index = []
    for ds in val_list[1:]:  # task 0 = firearm (intended leak), excluded
        for path, _ in ds.samples:
            wnid = Path(path).parent.name
            index.append((path, wnid_to_imagenet_idx[wnid]))
    return index


class BackgroundValDataset(torch.utils.data.Dataset):
    """ITIT per-image val set over ALL background images (M9), yielding the same
    item shape as PrivacyTaskCoalitionDataset so evaluate_leakage consumes it unchanged:
      (images (1, C, H, W), firearm_target=0, imagenet_targets (1,))
    """
    def __init__(self, index, loader, transform):
        self.index = index            # list of (path, imagenet_idx)
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        path, idx = self.index[i]
        img = self.transform(self.loader(path))
        return img.unsqueeze(0), 0, torch.tensor([idx], dtype=torch.long)


def load_backbone(args, device):
    ctor = getattr(models, args.arch)
    backbone = ctor(pretrained=False, gt=True, phase=args.phase)
    ckpt = torch.load(args.stage_b_ckpt, map_location="cpu", weights_only=False)
    coded_pwr = float(ckpt.get("coded_pwr", 1.0))
    state = ckpt["state_dict_backbone"] if "state_dict_backbone" in ckpt else ckpt["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    backbone.load_state_dict(state)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone, coded_pwr


def train_fresh_adversary(backbone, adversary, train_dataset, args, device, log, coded_pwr=1.0):
    snr_noise = snr_to_noise_std(args.SNR, args.GT_alg, coded_pwr,
                                 feat_channels=backbone.layer3[0].conv1.in_channels)
    optim = torch.optim.SGD(adversary.parameters(), lr=args.adv_lr, momentum=args.momentum,
                            weight_decay=args.weight_decay)
    for epoch in range(args.stage_c_epochs):
        loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True,
        )
        adversary.train()
        t0 = time.time()
        for it, (images, _, imagenet_target_per_image) in enumerate(loader):
            images = images.to(device); imagenet_target_per_image = imagenet_target_per_image.to(device)
            with torch.no_grad():
                pre = backbone.encode(images)
                post, _, _ = backbone.channel(pre, noise_std=snr_noise,
                                              gpu=device.index if device.type == "cuda" else None)
            if args.GT_alg == 1:
                adv_target = imagenet_target_per_image.reshape(-1)
                logits = adversary(post)
                loss = F.cross_entropy(logits, adv_target)
            else:
                num_classes = adversary.fc.out_features
                khot = torch.zeros(images.size(0), num_classes, device=device).scatter_(
                    1, imagenet_target_per_image, 1.0)
                logits = adversary(post)
                loss = F.binary_cross_entropy_with_logits(logits, khot)
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
            if it % args.print_freq == 0:
                line = f"[StageC][ep {epoch}][it {it:5d}] adv_loss={loss.item():.4f}"
                print(line); log.write(line + "\n"); log.flush()
        line = f"[StageC][ep {epoch}] time={time.time()-t0:.1f}s"
        print(line); log.write(line + "\n"); log.flush()


def evaluate_leakage(backbone, adversary, val_dataset, args, device, coded_pwr=1.0):
    """Returns dict with leakage metrics on val set."""
    snr_noise = snr_to_noise_std(args.SNR, args.GT_alg, coded_pwr,
                                 feat_channels=backbone.layer3[0].conv1.in_channels)
    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.val_workers, pin_memory=True, drop_last=False,
    )
    adversary.eval()
    correct = total = 0
    all_logits, all_targets = [], []
    with torch.no_grad():
        for images, _, imagenet_target_per_image in loader:
            images = images.to(device); imagenet_target_per_image = imagenet_target_per_image.to(device)
            pre = backbone.encode(images)
            post, _, _ = backbone.channel(pre, noise_std=snr_noise,
                                          gpu=device.index if device.type == "cuda" else None)
            if args.GT_alg == 1:
                adv_target = imagenet_target_per_image.reshape(-1)
                logits = adversary(post)
                pred = logits.argmax(dim=-1)
                correct += (pred == adv_target).sum().item()
                total += adv_target.numel()
            else:
                logits = adversary(post)
                all_logits.append(logits.cpu().numpy())
                num_classes = adversary.fc.out_features
                khot = torch.zeros(images.size(0), num_classes, device=device).scatter_(
                    1, imagenet_target_per_image, 1.0)
                all_targets.append(khot.cpu().numpy())

    if args.GT_alg == 1:
        return {"top1_imagenet_acc": correct / max(total, 1)}

    # GTGT-FM: per-class AUC + mAP
    from sklearn.metrics import roc_auc_score, average_precision_score
    logits = np.concatenate(all_logits, axis=0); targets = np.concatenate(all_targets, axis=0)
    aucs, aps = [], []
    for k in range(targets.shape[1]):
        if targets[:, k].sum() == 0:
            continue  # class never appeared in val
        aucs.append(roc_auc_score(targets[:, k], logits[:, k]))
        aps.append(average_precision_score(targets[:, k], logits[:, k]))
    return {"mean_auc": float(np.mean(aucs)), "mean_ap": float(np.mean(aps)),
            "num_classes_evaluated": len(aucs)}


def main():
    args = get_parser().parse_args()
    assert not (args.GT_alg == 1 and args.background_K != 0), \
        "ITIT (GT-alg 1) requires --background-K 0"
    if args.seed is not None:
        torch.manual_seed(args.seed)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log = open(os.path.join(args.output_dir, args.log_name), "w")
    log.write(f"args: {json.dumps(vars(args), indent=2)}\n"); log.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, coded_pwr = load_backbone(args, device)
    train_list, val_list = build_datasets(args)

    # Build a unified wnid mapping from both train and val to avoid label-space divergence (I2).
    all_wnids = sorted(set().union(*(ds.class_to_idx.keys() for ds in train_list + val_list)))
    wnid_to_imagenet_idx = {w: i for i, w in enumerate(all_wnids)}

    train_dataset = PrivacyTaskCoalitionDataset(train_list, args, split="train",
                                                wnid_to_imagenet_idx=wnid_to_imagenet_idx)
    val_dataset = PrivacyTaskCoalitionDataset(val_list, args, split="val",
                                              wnid_to_imagenet_idx=wnid_to_imagenet_idx)

    adversary = AdversaryHead(arch_name=args.arch,
                              num_classes=train_dataset.num_imagenet_classes,
                              pretrained=(args.adv_init == "pretrained")).to(device)

    train_fresh_adversary(backbone, adversary, train_dataset, args, device, log, coded_pwr=coded_pwr)

    if args.GT_alg == 1:
        bg_index = build_background_val_index(val_list, wnid_to_imagenet_idx)
        leakage_val = BackgroundValDataset(bg_index, val_list[1].loader, val_list[1].transform)
    else:
        leakage_val = val_dataset

    metrics = evaluate_leakage(backbone, adversary, leakage_val, args, device, coded_pwr=coded_pwr)
    print("Leakage metrics:", json.dumps(metrics, indent=2))
    log.write("leakage: " + json.dumps(metrics) + "\n"); log.flush()
    with open(os.path.join(args.output_dir, "leakage.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log.close()


if __name__ == "__main__":
    main()
