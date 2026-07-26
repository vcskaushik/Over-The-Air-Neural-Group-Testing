# HANDOFF — V1 Refinement (map the frontier before V2)

> **STATUS: ✅ COMPLETE (2026-07-26, commit `04bddf3`).** Full frontier + whole-sweep worst-case + BEST(λ_H=30) refresh done; results in `docs/superpowers/results/2026-07-25-...md` §9; cross-campaign summary in `docs/EXPERIMENT-LOG.md`.
> **Verdict:** confirms "no free-lunch" — a smooth entanglement-wall frontier; the recall-preserving privacy is modest (BEST cuts worst-case 27.8%→18.6%, only 1.5×) **and** not durable (rebounds to 22.3% after a refresh). The Kaiming attacker understated leakage up to ~18×.
> **→ Next steps (V2) are at the bottom of this file.** The "What to run / What to record" sections below are the now-executed playbook, kept for provenance.

**To:** the next Claude instance on the GPU box.
**Branch:** `dev/invariance-privacy-hsic-v1`.
**Prereq reading:** `HANDOFF.md` (base mechanics) and `docs/superpowers/results/2026-07-25-invariance-hsic-resnet18-results.md` (the first sweep + §9 refinement).

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
- [x] Fine grid {20,30,50,70} trained + Kaiming + pretrained-adv Stage C.
- [x] Pretrained-adv Stage C for the whole sweep (coarse + fine + baseline).
- [x] BEST (λ_H=30) refresh-tested, incl. pretrained-adv on the refreshed checkpoint.
- [x] Results doc updated with the full frontier + a clear verdict on "no free-lunch".
- [x] Committed + pushed (`04bddf3`).

The one sentence that matters: **is there a λ_H that meaningfully cuts worst-case leakage without missing firearms — and does it survive the refresh?**
**Answer: No.** Smooth entanglement-wall frontier; BEST=λ_H 30 cuts worst-case only 27.8%→18.6% at recall 48/50, and a 2-epoch refresh rebounds it to 22.3% (recall recovers to 50/50 but FPR triples to 4.02%). → **V2 needs a capacity lever, not more noise.**

---

## Next steps (V2) — bend the frontier with capacity

**Thesis to test:** the ResNet-18/ITIT entanglement wall is a *capacity* limit — class identity and the firearm decision are crammed into the same 128-ch layer-2 code. A **wider transmitted code** may let them separate, buying a given worst-case-leakage reduction at *less* recall cost, and holding it under refresh. If a bigger backbone can't bend it either, the honest conclusion is that firearm-vs-class isn't separable in this feature, and the privacy claim should be scoped accordingly.

### V2.1 — Primary experiment: ResNeXt-101 capacity sweep
- **Arch:** `-a resnext101_32x8d` (already wired in `resnet_design2/my_resnet.py`; 86.7M params; **layer-2 code = 512 ch, 4× ResNet-18's 128**). This is the one lever changed — keep ITIT, σ=0, K=0, same PK sampler, same Stage-C protocol, so it's directly comparable to §9.
- **Playbook:** mirror `run_hsic_refine.sh` on the new arch — Stage A pretrain (use ImageNet-pretrained ResNeXt-101, weights URL already in `my_resnet.py`, to skip a multi-day backbone pretrain) → λ_H sweep → utility(48k) → **pretrained-adv (worst-case) Stage C for every point** (make worst-case primary from the start; Kaiming optional) → pick BEST → refresh-test incl. worst-case on the refreshed checkpoint. Reuse `parse_results_v1refine.py` (arch-agnostic paths need a small tweak) for the table.
- **λ_H grid:** the HSIC scale may shift with the wider code — start `{0, 10, 30, 100, 300, 1000}`, then fill the knee once you see where recall breaks (as we did here).
- **The decision metric:** overlay the ResNeXt-101 utility-vs-worst-case-leakage frontier on the ResNet-18 one (§9.3). **Success = the curve bends down-and-left** (lower worst-case leakage at the *same* recall, e.g. a recall-preserving point ≥48/50 with worst-case < ~15% and refresh-durable). **Null = same proportional wall** at higher capacity ⇒ entanglement is intrinsic, not a capacity artifact.

### V2.2 — Secondary knobs (only after V2.1)
- **Channel noise σ>0 / VIB** (the other v1-deferred axis): does noise reduce leakage *independently* of the utility cost, or just add a second knob on the same wall? Pair with the capacity result.
- **Tap point / K>0 coalitions:** test whether the entanglement is specific to the ITIT layer-2 tap or the K=0 single-image regime (K>0 uses real group-testing coalitions and 5-D grouped inputs — also the case that would actually use >40 GB VRAM).

### V2.3 — Hardware / logistics
- **VRAM is a non-issue** for `resnext101_32x8d` at batch 32 (~15–25 GB est.; A40 = 44 GB). The binding constraint is **wall-clock**: ResNeXt-101 is ~9× ResNet-18 FLOPs → a comparable full sweep ≈ **~2 days on the A40**.
- **GPU recommendation:** **L40S** for best $/throughput (~½–⅓ the wall-clock; use AMP/bf16 for a further ~1.5–2×), or **A100** for max speed (HBM bandwidth helps grouped convs) / if moving to K>0. Don't stay on the A40 for the full V2 sweep.
- **Data:** the rebuilt dataset lives on mfs (1.28M small files). On a faster GPU it can become I/O-bound — stage `data/ImageNet-ILSVRC2012` on local NVMe if available and bump `-j`. Rebuild helpers (if the volume is wiped again): `pget.sh` + `fetch_{train,val}.sh` + `extract_train.sh` + `drain_inner.sh` (see memory `imagenet-rebuild-gotchas` for the mfs-quota / `--no-same-owner` / dedup pitfalls).

### V2 done when
- [ ] ResNeXt-101 frontier (utility vs **worst-case** leakage) measured across λ_H, refresh-tested at its BEST.
- [ ] Overlaid on the ResNet-18 frontier with a plain verdict: **does capacity bend the wall, yes or no?**
- [ ] `EXPERIMENT-LOG.md` + a new dated results doc updated; committed + pushed.
