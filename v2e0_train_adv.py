"""V2-E0 helper: retrain the sigma=0 worst-case (pretrained-adv) Stage-C adversary
on the frozen Stage-A encoder and SAVE its weights (eval_privacy.py trains one but
only persists leakage.json). The saved adversary is the frozen class probe used by
the Step-1 gate margin/collapse-SNR analysis. Also re-verifies leakage (~27.8%).

Run: .venv/bin/python v2e0_train_adv.py
"""
import argparse, json, os, torch
from types import SimpleNamespace
import resnet_design2 as models
from privacy.adversary import AdversaryHead
from privacy.dataset import PrivacyTaskCoalitionDataset
from privacy.eval_privacy import (build_datasets, load_backbone, train_fresh_adversary,
                                  evaluate_leakage, build_background_val_index, BackgroundValDataset)

OUT = "Trained_Models/StageC_OnStageA_ResNet18_advpretrained"

args = SimpleNamespace(
    data="data/GroupTestingDataset", task_num=2, background_K=0, GT_alg=1,
    arch="resnet18", phase=False, SNR=None,
    stage_b_ckpt="Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar",
    adv_init="pretrained", stage_c_epochs=30, adv_lr=1e-3, momentum=0.9,
    weight_decay=1e-4, batch_size=32, workers=8, val_workers=4,
    output_dir=OUT, log_name="eval_c_retrain.log", print_freq=100, seed=0,
)
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(args.seed)
device = torch.device("cuda")
log = open(os.path.join(OUT, args.log_name), "w")

backbone, coded_pwr = load_backbone(args, device)
print(f"coded_pwr={coded_pwr}")
train_list, val_list = build_datasets(args)
all_wnids = sorted(set().union(*(ds.class_to_idx.keys() for ds in train_list + val_list)))
wnid_to_imagenet_idx = {w: i for i, w in enumerate(all_wnids)}
train_ds = PrivacyTaskCoalitionDataset(train_list, args, split="train", wnid_to_imagenet_idx=wnid_to_imagenet_idx)
print(f"num_imagenet_classes={train_ds.num_imagenet_classes}")

adversary = AdversaryHead(arch_name=args.arch, num_classes=train_ds.num_imagenet_classes,
                          pretrained=True).to(device)
train_fresh_adversary(backbone, adversary, train_ds, args, device, log, coded_pwr=coded_pwr)

torch.save({"state_dict": adversary.state_dict(),
            "num_classes": train_ds.num_imagenet_classes,
            "coded_pwr": coded_pwr, "arch": args.arch},
           os.path.join(OUT, "adversary.pth"))
print("saved adversary.pth")

bg_index = build_background_val_index(val_list, wnid_to_imagenet_idx)
leak_val = BackgroundValDataset(bg_index, val_list[1].loader, val_list[1].transform)
m = evaluate_leakage(backbone, adversary, leak_val, args, device, coded_pwr=coded_pwr)
print("re-verified leakage:", json.dumps(m))
log.write("leakage_retrain: " + json.dumps(m) + "\n"); log.close()
print("V2E0_ADV_DONE")
