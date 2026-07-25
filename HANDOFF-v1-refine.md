# HANDOFF — V1 Refinement (map the frontier before V2)

**To:** the next Claude instance on the GPU box.
**Branch:** `dev/invariance-privacy-hsic-v1`.
**Prereq reading:** `HANDOFF.md` (base mechanics) and `docs/superpowers/results/2026-07-25-invariance-hsic-resnet18-results.md` (the first sweep).

## Why
The first HSIC sweep validated the thesis (privacy survives the refresh, unlike the adversarial predecessor), but has two gaps that must close before we scope V2:
1. **The λ_H grid skipped the entire 10→100 knee**, so "no free-lunch λ_H" is asserted from a grid that jumps over the interesting region. Need a **fine grid in [10,100]**.
2. **Worst-case (pretrained-adv) leakage was measured for only 3 checkpoints**, while the headline led with the weak Kaiming attacker (which understates ~15×). Need **pretrained-adv Stage C across the whole sweep** — it is the primary metric.

## What to run
Everything is scripted. Order:

### 1. Environment + checkpoints + dataset
- **`.venv`**: recreate a CUDA venv (`.venv/bin/python`) as in `HANDOFF.md` §1 (the sweep scripts hardcode `PY=.venv/bin/python`). The first run used torch 2.8.0+cu128; any recent CUDA torch + torchvision + numpy + scikit-learn + pytest is fine. Verify: `.venv/bin/python -m pytest tests/ -q -m "not slow and not gpu"` → 49 passed.
- **Restore checkpoints from Drive** (avoids redoing Stage A + the 5 coarse Stage-B runs). Bootstrap's `/workspace` restore should bring `Trained_Models/` back; verify it landed:
  ```bash
  ls Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar \
     Trained_Models/Invariance_ResNet18_lam{0,1,10,100,1000}/invariance_final.pth.tar
  ```
  If missing, pull them explicitly:
  ```bash
  rclone copy gdrive:vast-backup/workspace/Over-The-Air-Neural-Group-Testing/Trained_Models Trained_Models --progress
  ```
- **Rebuild the dataset** (it is sync-excluded, so it is gone). Use the same scripts as the first run — see `docs/superpowers/results/2026-07-25-*.md` §1 provenance:
  ```bash
  .venv/bin/python data_scripts/materialize_imagenet_from_hf.py train --workers 8   # ~hours, resumable
  # val needs original filenames: download ILSVRC2012_img_val.tar (image-net.org) to data/, then:
  .venv/bin/python data_scripts/valprep_canonical.py
  # reorganize into GroupTestingDataset (now uses symlinks, non-destructive):
  cd data && python ../data_scripts/create_dataset_from_imagenet.py > create_dataset.sh && sh create_dataset.sh && cd ..
  ```
  Verify: `ls data/GroupTestingDataset/1/train | wc -l` → 976; firearm val files resolve.

### 2. Fine grid + worst-case sweep (one script)
```bash
bash run_hsic_refine.sh
```
This (idempotent, skip-if-exists):
- Trains λ_H ∈ **{20, 30, 50, 70}** → utility(48k) → Kaiming Stage C.
- Runs **`--adv-init pretrained` Stage C on the WHOLE sweep**: baseline + λ_H ∈ {0,1,10,20,30,50,70,100,1000} (reuses the two already computed).

### 3. Pick the sweet spot, then refresh-test it
From the fine-grid results, **BEST = the largest worst-case (pretrained-adv) leakage drop while firearm recall stays ≥ ~48/50** (AUC ≥ ~0.99). If no point holds recall while dropping leakage, that *confirms* the entanglement wall — report it.
```bash
bash run_refresh_and_worstcase.sh <BEST>          # 2-epoch refresh + Kaiming Stage C + pretrained Stage C on unrefreshed BEST
# ALSO run pretrained-adv Stage C on the REFRESHED BEST (the key refresh-durability worst-case number):
.venv/bin/python -u -m privacy.eval_privacy \
    --stage-b-ckpt Trained_Models/RefreshedInvariance_lam<BEST>/checkpoint.pth.tar \
    --data data/GroupTestingDataset --task-num 2 --background-K 0 --GT-alg 1 -a resnet18 \
    --stage-c-epochs 30 --batch-size 32 --adv-init pretrained -j 8 -valj 4 \
    --output_dir Trained_Models/StageC_RefreshedInvariance_lam<BEST>_advpretrained
```
Optionally bisect once more around BEST if the knee is sharp.

## What to record
**Append a "V1 refinement" section to the existing results doc** `docs/superpowers/results/2026-07-25-invariance-hsic-resnet18-results.md` (don't start a new file). Add:
1. **The full frontier table** — every λ_H (coarse + fine), columns: utility Acc@1 / ROC-AUC / firearm recall / false-pos, **Kaiming leakage AND pretrained-adv (worst-case) leakage**. Make pretrained-adv the emphasized column.
2. **A utility-vs-worst-case-leakage frontier** (the Pareto view): does a λ_H exist with real leakage reduction at recall ≥ ~48/50, or is the knee a cliff (entanglement wall)?
3. **Refresh durability at BEST** — worst-case leakage before/after refresh, and utility (recall + FP count) before/after.
4. **Revised bottom line**: does the fine grid overturn or confirm "no free-lunch"? This is the input that decides V2's scope (ResNet-18+noise vs. a capacity lever like ResNeXt-101 / GTGT-FM).

Use `parse_results.py` to scrape the `leakage.json` / utility logs into the table. Commit + push the updated results doc (doc + scripts only; checkpoints/logs are gitignored and auto-synced to Drive).

## Done when
- [ ] Fine grid {20,30,50,70} trained + Kaiming + pretrained-adv Stage C.
- [ ] Pretrained-adv Stage C for the whole sweep (coarse + fine + baseline).
- [ ] BEST refresh-tested, incl. pretrained-adv on the refreshed checkpoint.
- [ ] Results doc updated with the full frontier + a clear verdict on "no free-lunch".
- [ ] Committed + pushed.

The one sentence that matters: **is there a λ_H that meaningfully cuts worst-case leakage without missing firearms — and does it survive the refresh?** That decides whether V2 stays on ResNet-18+noise or needs a capacity lever.
