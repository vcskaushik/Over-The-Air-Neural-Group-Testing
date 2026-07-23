"""Stage-B replacement: joint E+R HSIC invariance training (ITIT, sigma=0, v1).

No learned adversary. Warm-starts E+R from a Stage-A checkpoint and trains both
on L_task + lambda_H * HSIC(background). See the design spec for scope/risks.
"""
import argparse
import json
import os
import pathlib
import time

import torch

import resnet_design2 as models
from privacy.train_privacy import build_datasets, load_stage_a, validate_utility
from privacy.dataset import PrivacyTaskCoalitionDataset
from privacy.sampler import PKBackgroundSampler
from privacy.hsic import HSICPenalty, delta_label_kernel, rbf_kernel, median_bandwidth, hsic_biased
from privacy.trainer import joint_step


def get_parser():
    p = argparse.ArgumentParser("OTA-NGT HSIC invariance training (v1)")
    p.add_argument("--data", required=True)
    p.add_argument("--task-num", type=int, default=2)
    p.add_argument("--background-K", type=int, required=True)
    p.add_argument("--GT-alg", type=int, choices=[1, 2], required=True)
    p.add_argument("-a", "--arch", required=True)
    p.add_argument("--phase", action="store_true")
    p.add_argument("--SNR", type=float, default=None)
    p.add_argument("--stage-a-ckpt", required=True)
    p.add_argument("--hsic-lambda", type=float, default=1.0)
    p.add_argument("--hsic-extractor", choices=["pretrained", "random", "both"], default="pretrained")
    p.add_argument("--hsic-num-random", type=int, default=2)
    p.add_argument("--hsic-bandwidths", default="0.5,1,2")
    p.add_argument("--pk-classes", type=int, default=8)
    p.add_argument("--pk-per-class", type=int, default=4)
    p.add_argument("--pk-firearm", type=int, default=4)
    p.add_argument("--enc-lr", type=float, default=1e-4)
    p.add_argument("--rec-lr", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)  # unused with batch_sampler; kept for val
    p.add_argument("-j", "--workers", type=int, default=8)
    p.add_argument("-valj", "--val-workers", type=int, default=4)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--log-name", default="invariance.log")
    p.add_argument("--print-freq", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    return p


def save_invariance_ckpt(backbone, coded_pwr, epoch, path):
    state = {f"module.{k}": v for k, v in backbone.state_dict().items()}
    torch.save({"state_dict": state, "coded_pwr": float(coded_pwr),
                "epoch": int(epoch), "arch": getattr(backbone, "arch_name", "")}, path)


def extractor_sanity(backbone, hsic_penalty, loader, device, snr_noise, num_batches=3):
    """HSIC(Phi(z), c) on true vs permuted labels — should be clearly higher for true."""
    backbone.eval()
    true_vals, perm_vals = [], []
    with torch.no_grad():
        for bi, (images, firearm_target, imagenet_target) in enumerate(loader):
            if bi >= num_batches:
                break
            images = images.to(device)
            pre = backbone.encode(images)
            post, _, _ = backbone.channel(pre, noise_std=snr_noise,
                                          gpu=device.index if device.type == "cuda" else None)
            bg = (firearm_target == 0)
            if int(bg.sum()) < 2:
                continue
            feats = post[bg]
            labels = imagenet_target[bg].reshape(-1).to(device)
            true_vals.append(float(hsic_penalty(feats, labels)))
            perm = labels[torch.randperm(labels.numel(), device=device)]
            perm_vals.append(float(hsic_penalty(feats, perm)))
    avg = lambda xs: sum(xs) / max(len(xs), 1)
    return avg(true_vals), avg(perm_vals)


def joint_loop(backbone, hsic_penalty, train_dataset, val_dataset, args, device, log, coded_pwr):
    enc_params = list(backbone.conv1.parameters()) + list(backbone.bn1.parameters()) \
        + list(backbone.layer1.parameters()) + list(backbone.layer2.parameters())
    rec_params = list(backbone.layer3.parameters()) + list(backbone.layer4.parameters()) \
        + list(backbone.fc.parameters())
    optimizer = torch.optim.SGD(
        [{"params": enc_params, "lr": args.enc_lr},
         {"params": rec_params, "lr": args.rec_lr}],
        momentum=args.momentum, weight_decay=args.weight_decay)

    snr_noise = None  # v1: sigma = 0 (asserted in main)

    sampler = PKBackgroundSampler.from_dataset(
        train_dataset, args.pk_classes, args.pk_per_class, args.pk_firearm,
        seed=(args.seed or 0))

    # Startup diagnostic (M6): warn if the frozen extractor is inert on real features.
    diag_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=PKBackgroundSampler.from_dataset(
            train_dataset, args.pk_classes, args.pk_per_class, args.pk_firearm, seed=999),
        num_workers=args.workers, pin_memory=True)
    t_true, t_perm = extractor_sanity(backbone, hsic_penalty, diag_loader, device, snr_noise)
    line = f"[Diag] extractor HSIC true={t_true:.3e} permuted={t_perm:.3e}"
    print(line); log.write(line + "\n"); log.flush()
    if t_true <= t_perm:
        line = "[Diag][WARN] extractor HSIC not above permuted-label baseline — extractor may be inert!"
        print(line); log.write(line + "\n"); log.flush()

    for epoch in range(args.epochs):
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=PKBackgroundSampler.from_dataset(
                train_dataset, args.pk_classes, args.pk_per_class, args.pk_firearm,
                seed=(args.seed or 0) + epoch),
            num_workers=args.workers, pin_memory=True)
        t0 = time.time()
        for it, (images, firearm_target, imagenet_target_per_image) in enumerate(loader):
            metrics = joint_step(
                backbone=backbone, hsic_penalty=hsic_penalty, optimizer=optimizer,
                images=images, firearm_target=firearm_target,
                imagenet_target_per_image=imagenet_target_per_image,
                hsic_lambda=args.hsic_lambda, device=device, snr_noise_std=snr_noise)
            if it % args.print_freq == 0:
                line = (f"[Invar][ep {epoch}][it {it:4d}] util={metrics['loss_util']:.4f} "
                        f"hsic={metrics['loss_hsic']:.3e} total={metrics['loss_total']:.4f}")
                print(line); log.write(line + "\n"); log.flush()
        acc = validate_utility(backbone, val_dataset, args, device, coded_pwr=coded_pwr)
        line = f"[Invar][ep {epoch}] val_firearm_acc={acc:.4f} time={time.time()-t0:.1f}s"
        print(line); log.write(line + "\n"); log.flush()


def main():
    args = get_parser().parse_args()
    assert args.GT_alg == 1 and args.background_K == 0, "v1 is ITIT-only (--GT-alg 1 --background-K 0)"
    assert args.SNR is None, "v1 is sigma=0; --SNR belongs to v2"
    if args.seed is not None:
        torch.manual_seed(args.seed)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log = open(os.path.join(args.output_dir, args.log_name), "w")
    log.write(f"args: {json.dumps(vars(args), indent=2)}\n"); log.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, coded_pwr = load_stage_a(args, device)
    backbone.arch_name = args.arch

    train_list, val_list = build_datasets(args)
    all_wnids = sorted(set().union(*(ds.class_to_idx.keys() for ds in train_list + val_list)))
    wnid_to_imagenet_idx = {w: i for i, w in enumerate(all_wnids)}

    train_dataset = PrivacyTaskCoalitionDataset(train_list, args, split="train",
                                                wnid_to_imagenet_idx=wnid_to_imagenet_idx,
                                                invariance_mode=True)
    val_dataset = PrivacyTaskCoalitionDataset(val_list, args, split="val",
                                              wnid_to_imagenet_idx=wnid_to_imagenet_idx)

    bandwidths = tuple(float(x) for x in args.hsic_bandwidths.split(","))
    hsic_penalty = HSICPenalty(arch_name=args.arch, extractor=args.hsic_extractor,
                               num_random=args.hsic_num_random, bandwidth_mults=bandwidths).to(device)

    joint_loop(backbone, hsic_penalty, train_dataset, val_dataset, args, device, log, coded_pwr)
    save_invariance_ckpt(backbone, coded_pwr, args.epochs,
                         os.path.join(args.output_dir, "invariance_final.pth.tar"))
    log.close()


if __name__ == "__main__":
    main()
