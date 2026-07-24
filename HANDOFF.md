# HANDOFF — Run the HSIC Invariance v1 Experiments (ResNet-18)

**To:** the next Claude instance (running on the GPU box, e.g. an A40).
**From:** the instance that designed, implemented, and reviewed the HSIC invariance training path.
**Date:** 2026-07-24
**Branch:** `dev/invariance-privacy-hsic-v1` (HEAD `10545e8`). All code is committed and tested; **no training has been run yet.**

Your job, in order:
1. **Download + create the dataset** (§2)
2. **Run all ResNet-18 training + evaluation** (§3)
3. **Record all results** in a results doc and commit (§4)

Do not change the mechanism design. If something in the code looks wrong, prefer reporting it in your results doc over silently altering the training path — the code passed a full spec + whole-branch review. Read §5 (gotchas) before you start.

---

## 0. What this is (read first)

OTA-NGT splits a ResNet across a noisy channel: encoder `E` (conv1..layer2) transmits post-channel features; receiver `R` (layer3..fc) does binary **firearm-vs-background** detection. An eavesdropper taps the same features to recover the fine-grained ImageNet class. **v1 goal:** train `E+R` jointly so the transmitted features are statistically **independent** of the fine-grained class (HSIC penalty) while keeping firearm utility — replacing an earlier adversarial approach that failed (privacy "gain" was obfuscation that a retrained adversary/refresh recovered).

**v1 scope (do not deviate):** HSIC only, **ITIT only** (`--GT-alg 1 --background-K 0`), **σ=0 (no channel noise — do NOT pass `--SNR`).** VIB and σ>0 are v2.

**Required reading before you run anything:**
- Spec: `docs/superpowers/specs/2026-07-23-invariance-privacy-hsic-vib-design.md`
- Plan: `docs/superpowers/plans/2026-07-23-invariance-privacy-hsic.md`
- First-pass results (the failed adversarial approach; **use as the format template for your results doc**): `docs/superpowers/results/2026-04-18-itit-resnet18-no-noise-results.md`
- Conceptual framing + known risks R1–R5: the `[Temporary — Working Notes]` section in `Privacy_Preserving_OTA_NGT.tex`
- Deferred minor findings + the whole SDD build log: `.superpowers/sdd/progress.md`

**Key code:** `privacy/train_invariance.py` (entry point), `privacy/hsic.py` (HSIC penalty + frozen extractors), `privacy/sampler.py` (PK sampler), `privacy/trainer.py::joint_step`, `privacy/eval_privacy.py` (Stage C).

---

## 1. Environment setup

This box needs a **CUDA** Python env with torch + torchvision + numpy + scikit-learn + pytest. (There may be a local CPU-only `.venv` from the design session — recreate it for GPU.)

```bash
cd <repo-root>
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# install the CUDA build that matches this box's driver (example: cu121)
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install numpy scikit-learn pytest tensorboard
```

Verify the code is intact before running experiments:
```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect True
.venv/bin/python -m pytest tests/ -q -m "not slow and not gpu"                             # expect 49 passed
```
If the 49 offline tests do not pass, stop and report — the environment is broken, not the plan.

---

## 2. Task 1 — Download and create the dataset

The dataset is essentially **ImageNet-1k minus ~21 classes**, reorganized into `data/GroupTestingDataset/{0,1}/{train,val}/<wnid>/*.JPEG` where **task 0 = 3 firearm classes**, **task 1 = 976 background classes**. Budget **~140 GB** on the volume (see below).

### 2a. Get ImageNet ILSVRC2012
You need the standard `ILSVRC2012_img_train.tar` (~138 GB) and `ILSVRC2012_img_val.tar` (~6.3 GB), extracted into per-class folders:
```
data/ImageNet-ILSVRC2012/train/<wnid>/*.JPEG
data/ImageNet-ILSVRC2012/val/<wnid>/*.JPEG
```
- If this box already has ImageNet mounted, point at it (or symlink) and skip the download.
- Otherwise download from your institutional/Kaggle/HF mirror, then extract. Helper scripts exist: `data_scripts/extract_ILSVRC.sh` / `extract_ILSVRC2.sh` (train tar → per-class tars → extract; the val set must be reorganized into `<wnid>/` folders using the ILSVRC2012 val ground-truth — the standard "valprep" step).
- **Volume sizing:** ~140 GB steady state; **~300 GB if you download+extract on the same volume** (tarballs + extracted coexist transiently — delete tarballs after extraction). If ImageNet is already extracted elsewhere, ~30 GB is enough (symlink).

### 2b. Build GroupTestingDataset
The generator script `data_scripts/create_dataset_from_imagenet.py` reads relative paths `ImageNet-ILSVRC2012` and `GroupTestingDataset`, so run it **from inside `data/`**:
```bash
cd data
python ../data_scripts/create_dataset_from_imagenet.py > create_dataset.sh
# INSPECT create_dataset.sh FIRST — see the warning below — then:
sh create_dataset.sh
cd ..
```

**⚠️ Critical warning:** the generated script uses **`mv`**, which **moves (destroys) the ImageNet source** into `GroupTestingDataset`, and starts with `rm -rf GroupTestingDataset`. If you need to keep the original ImageNet tree (or it's a shared/read-only mount), edit `data_scripts/create_dataset_from_imagenet.py` to emit `cp -r` or `ln -s` instead of `mv` (there are commented-out `ln -s` variants in the script) before generating the shell script.

The firearm wnids are `n02749479` (assault rifle), `n04086273` (revolver), `n04090263` (rifle); 21 "ban" classes (holster, cannon, military uniform, etc.) are excluded. Expect the script to print `number of non-firearm classes: 976`.

### 2c. Verify the dataset
```bash
ls data/GroupTestingDataset/            # -> 0  1
ls data/GroupTestingDataset/0/train | wc -l    # -> 3   (firearm classes)
ls data/GroupTestingDataset/1/train | wc -l    # -> 976 (background classes)
ls data/GroupTestingDataset/0/val data/GroupTestingDataset/1/val >/dev/null && echo "val OK"
```
If these counts are off, the dataset is wrong — fix before training. Everything downstream (`--data data/GroupTestingDataset`) depends on this layout.

---

## 3. Task 2 — Training + evaluation on ResNet-18

Run everything from the repo root with `.venv/bin/python`. Prefix GPU selection with `CUDA_VISIBLE_DEVICES=0`. Checkpoints land under `Trained_Models/` (gitignored). Rough time budget on an A40: Stage A ~25 min; each invariance run ~10–20 min; each Stage C ~20–45 min; **full sweep ≈ half a day** (I/O-bound — see §5).

### 3a. Stage A — utility pretrain (skip if a `StageA_ITIT_ResNet18` checkpoint already exists)
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u main.py \
    --data data/GroupTestingDataset --task-num 2 --background-K 0 --GT-alg 1 \
    -a resnet18 --pretrained --epochs 20 --batch-size 32 --lr 0.001 \
    --output_dir Trained_Models/StageA_ITIT_ResNet18 --log-name stage_a.log
```
Best checkpoint: `Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar`. Sanity: Acc@1 should be ~98%, ROC-AUC ~1.0, firearm recall 50/50 (matches the first-pass doc).

### 3b. Baseline leakage — Stage C on the un-privatized Stage A encoder
This is the "no privacy" leakage reference (first pass got ~35%). The invariance checkpoints are measured against this.
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m privacy.eval_privacy \
    --stage-b-ckpt Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar \
    --data data/GroupTestingDataset --task-num 2 --background-K 0 --GT-alg 1 \
    -a resnet18 --stage-c-epochs 30 --batch-size 32 --adv-init kaiming \
    --output_dir Trained_Models/StageC_OnStageA_ResNet18
```
Leakage JSON → `Trained_Models/StageC_OnStageA_ResNet18/leakage.json` (`top1_imagenet_acc`). **Note:** Stage C now evaluates leakage over the **full background val set** (M9), so this number is directly comparable across configs.

### 3c. Stage B — HSIC invariance training (the λ_H sweep + mandatory control)
Run **`--hsic-lambda 0` first** — this is the **mandatory control** (joint E+R + PK sampler, no HSIC). Without it, every reported delta is confounded by the sampler's distribution shift (PK batches are ~11% firearm vs. Stage A's 50/50). Then sweep λ_H upward.

**λ_H values to run:** `0` (control), then `1, 10, 100, 1000`. Biased HSIC at m=32 is numerically small, so the default `1.0` is likely too weak to move anything — expect the interesting regime at the higher end; extend the sweep upward if even `1000` barely dents leakage, or downward if utility collapses early.

```bash
for LAM in 0 1 10 100 1000; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m privacy.train_invariance \
      --stage-a-ckpt Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar \
      --data data/GroupTestingDataset --task-num 2 --background-K 0 --GT-alg 1 \
      -a resnet18 \
      --hsic-lambda $LAM --hsic-extractor pretrained --hsic-num-random 2 \
      --pk-classes 8 --pk-per-class 4 --pk-firearm 4 \
      --epochs 30 --enc-lr 1e-4 --rec-lr 1e-4 --batch-size 32 -j 8 -valj 4 \
      --output_dir Trained_Models/Invariance_ResNet18_lam${LAM} --log-name invariance.log
done
```
Checkpoint per run: `Trained_Models/Invariance_ResNet18_lam${LAM}/invariance_final.pth.tar` (already main.py-`--resume` compatible).

**⚠️ Watch the startup diagnostic (M6).** Each run logs, before training:
`[Diag] extractor HSIC true=<x> permuted=<y>`. If you see `[Diag][WARN] extractor HSIC not above permuted-label baseline`, the frozen extractor is **inert on the real features** and the whole run optimizes nothing — **stop and report it** (likely a BN/distribution-mismatch issue on the pretrained extractor). **Record the true/permuted values for every run** in your results doc.

**A40 note:** you have VRAM to spare on ResNet-18. Consider a larger PK batch (`--pk-classes 16 --pk-per-class 8`) — bigger m makes the HSIC estimator less noisy (mitigates R2/R3). If you do, note it in the results.

### 3d. Utility of each invariance checkpoint (48k protocol, apples-to-apples)
The invariance checkpoint is already main.py-compatible, so evaluate utility directly (no conversion needed):
```bash
for LAM in 0 1 10 100 1000; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u main.py --evaluate \
      --resume Trained_Models/Invariance_ResNet18_lam${LAM}/invariance_final.pth.tar \
      --data data/GroupTestingDataset --task-num 2 --background-K 0 --GT-alg 1 \
      -a resnet18 --batch-size 32
done
```
Record Firearm Acc@1, ROC-AUC, firearm recall (x/50), and false-positive count from each.

### 3e. Stage C — honest leakage for each invariance checkpoint
```bash
for LAM in 0 1 10 100 1000; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m privacy.eval_privacy \
      --stage-b-ckpt Trained_Models/Invariance_ResNet18_lam${LAM}/invariance_final.pth.tar \
      --data data/GroupTestingDataset --task-num 2 --background-K 0 --GT-alg 1 \
      -a resnet18 --stage-c-epochs 30 --batch-size 32 --adv-init kaiming \
      --output_dir Trained_Models/StageC_Invariance_ResNet18_lam${LAM}
done
```
Also run **one** `--adv-init pretrained` arm on the strongest-privacy checkpoint as a worst-case attacker check.

### 3f. Refresh-fragility diagnostic (R4) — the test that killed v1
For the best non-trivial λ_H (and the λ_H=0 control), do a brief utility-only refresh, then re-run Stage C. **This is reported as a deployment-lifecycle fragility number, NOT the honest leakage.** The honest number is 3e on the unrefreshed encoder.
```bash
LAM=<best>
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u main.py \
    --resume Trained_Models/Invariance_ResNet18_lam${LAM}/invariance_final.pth.tar \
    --data data/GroupTestingDataset --task-num 2 --background-K 0 --GT-alg 1 \
    -a resnet18 --epochs 2 --batch-size 32 --lr 0.0001 \
    --output_dir Trained_Models/RefreshedInvariance_lam${LAM}
# then Stage C on Trained_Models/RefreshedInvariance_lam${LAM}/checkpoint.pth.tar (same eval_privacy command as 3e)
```
The v1 question this answers: **does the HSIC privacy survive a refresh** (unlike the old adversarial approach, whose gain fully rebounded)? That is the decisive result.

---

## 4. Task 3 — Record all results

Create `docs/superpowers/results/<YYYY-MM-DD>-invariance-hsic-resnet18-results.md`, **modeled on** `docs/superpowers/results/2026-04-18-itit-resnet18-no-noise-results.md` (same structure: headline table, setup, per-stage sections, caveats, interpretation, file pointers). Record **at minimum**:

1. **Setup:** exact hardware, torch/cuda versions, dataset image counts, all hyperparameters, PK batch shape, seed(s), how many seeds.
2. **Headline table** — one row per λ_H (incl. the `0` control), columns:
   - Utility Acc@1 (48k), ROC-AUC, firearm recall (x/50), false-positive count
   - **Stage-C leakage** = top-1 ImageNet acc on the full background val (chance ≈ 1/976 ≈ 0.1%)
   - vs. the **baseline** (Stage C on Stage A, §3b)
3. **The λ_H=0 control** called out explicitly — utility and leakage deltas attributable to sampling/joint-training alone, so the HSIC effect is isolated.
4. **Extractor-sanity diagnostic** (`[Diag] true/permuted`) for every run — and whether any WARN fired.
5. **Refresh diagnostic (§3f):** utility and leakage before vs. after refresh; state plainly whether the privacy survived.
6. **`--adv-init pretrained` worst-case** leakage on the strongest checkpoint.
7. **Trade-off read:** is there a λ_H that reduces leakage meaningfully below baseline while keeping firearm utility — and does it survive the refresh? That is the paper-relevant finding. If nothing beats the entanglement floor, say so — a clean negative is a real result here (the first pass was a negative).
8. **Caveats:** single-seed noise, Stage-C epoch count (30 vs spec's 60 — a longer Stage C may find more leakage; note it as a lower bound), I/O notes.

Keep the raw `leakage.json` files and logs under `Trained_Models/` (gitignored) and reference their paths in the doc. **Commit the results doc** (and only the doc + any small analysis scripts — not checkpoints/logs) to the branch.

```bash
git add docs/superpowers/results/<your-file>.md
git commit -m "docs: add HSIC invariance v1 results (ResNet-18, ITIT, sigma=0)"
```

---

## 5. Gotchas / watch-fors (read before running)

- **I/O, not the GPU, is the bottleneck.** ~1.3M small JPEGs on a network volume will starve the A40. Stage the dataset on **local NVMe scratch** if available; use `-j 8`–`16`. If `nvidia-smi` shows low/volatile GPU util, you're disk-bound — that's expected, not a bug.
- **The pretrained HSIC extractor downloads torchvision ImageNet weights on first use** — the box needs network access (or a pre-warmed `torch.hub`/`TORCH_HOME` cache). Air-gapped nodes will fail here.
- **σ=0 only.** Do **not** pass `--SNR` to `train_invariance` — it asserts `SNR is None` (v1). Noise/VIB is v2.
- **HSIC is a training signal only. Never report it as the privacy number** — Stage C is the arbiter. If the extractor-sanity WARN fires, HSIC-in-training may look great while leakage doesn't move (this is the R1/R3 failure mode); trust Stage C.
- **`--hsic-lambda 1.0` (the default) is a placeholder** — biased HSIC at m=32 is small, so it may be far too weak. Sweep upward; if utility never drops and leakage never moves across the whole sweep, suspect an inert extractor (M6 diagnostic) before concluding "HSIC does nothing."
- **Don't re-run completed work blindly.** `.superpowers/sdd/progress.md` records what the code build did; it is not a training log. There is no training ledger yet — you're creating the first results.
- **Deferred code nits** (non-blocking, from the final review, in `.superpowers/sdd/progress.md`): a weak assertion in `tests/test_hsic.py:75`, missing `hasattr(arch)` guards in `hsic.py`, an untested `<2-background` branch in `joint_step`, and a `val_list[1]` IndexError if `--task-num < 2` (don't run ITIT with `--task-num 1`). None affect the experiments if you follow the commands above.
- **If you must change the mechanism** to get a sensible result (e.g., the extractor is inert), do it as a **separate, tested, committed change** with the reason documented — do not quietly edit and re-run.

---

## 6. Definition of done

- [ ] Dataset built and verified (§2c counts correct).
- [ ] Stage A checkpoint exists (or reused) with sane utility.
- [ ] Baseline Stage-C leakage recorded.
- [ ] λ_H sweep **including the λ_H=0 control** trained; extractor-sanity diagnostic captured for each.
- [ ] Utility (48k) + Stage-C leakage recorded for every checkpoint.
- [ ] Refresh-fragility diagnostic run and reported.
- [ ] Results doc written (in the first-pass format), committed to `dev/invariance-privacy-hsic-v1`.
- [ ] A one-paragraph bottom-line: **did HSIC invariance reduce leakage below baseline at acceptable utility, and did it survive the refresh?**

Good luck. The single most important number is the **refreshed Stage-C leakage of the best λ_H vs. the baseline** — that's whether this approach beats the one it replaced.
