# HANDOFF — V2-E2: Does capacity bend the wall? (single operating-point gate)

> **The primary V2 experiment, run gated (E0 philosophy): one recall-preserving operating point decides the go/no-go before any full frontier sweep.** Capacity (ResNeXt-101, 4× wider code) is the one untested lever — channel noise was killed by Campaign D. Expected cost: a Stage-A fine-tune (once) + ~3 invariance runs + one 60-ep Stage-C + a free probe.

**To:** the next Claude instance on the GPU box.
**Branch:** `dev/invariance-privacy-hsic-v1`.
**Prereq reading:** `docs/superpowers/specs/2026-07-25-v2-capacity-noise-vib-design.md` §3 (the gated E2 design), `docs/superpowers/results/2026-07-25-invariance-hsic-resnet18-results.md` §9 (the ResNet-18 frontier being beaten), `docs/superpowers/results/2026-07-26-v2e0-snr-window-resnet18.md` §6 (why Stage-C is now 60-ep).

---

## The question, and the single number that answers it

V1's entanglement wall summarizes to one operating point: **worst-case leakage at the recall-preserving edge** (firearm recall 50/50, matched FPR≤2%, **60-ep converged** Stage-C). ResNet-18's value there is **~22%** (HSIC λ=30, 60-ep — this is already the converged comparator, so the comparison below is matched-convergence). The capacity question is:

> On ResNeXt-101 (512-ch code, 4× ResNet-18's 128), at recall = 50/50 (matched FPR≤2%, 60-ep Stage-C), is worst-case (pretrained-adv) leakage **meaningfully below ~22%** (beyond the ~1 pp cross-campaign band)?

- **Yes** → capacity bent the wall → expand to the full frontier + refresh-durability (spec §3 E2.2), then E3.
- **No (≈22%)** → the wall persists at 4× width → **stop the capacity campaign**; scope the privacy claim to "firearm and class are dimensionally inseparable in this feature," and pivot to the deeper open question (is the entanglement a layer-2-tap / K=0 artifact? — test alternate tap layers / K>0 next).

Do **not** run the full λ_H frontier for the go/no-go — one operating point suffices (this is the E0 gate philosophy).

---

## Prerequisites

```bash
cd <repo>/Over-The-Air-Neural-Group-Testing        # confirm path
git checkout dev/invariance-privacy-hsic-v1 && git pull
.venv/bin/python -m pytest tests/ -q -m "not slow and not gpu"     # expect 49 passed
ls data/GroupTestingDataset/1/train | wc -l        # 976; firearm val files resolve
```
- **This is σ=0 (no channel noise), so the `device.index=None` σ>0 crash and VIB are NOT needed here** — they're E3-only.
- **ResNeXt-101 is memory-heavier** (~88M params, 512-ch code). If batch 32 OOMs on the A40 (46 GB), drop to `--batch-size 16` (keep it consistent across Stage-A / invariance / Stage-C).
- **The ResNet-18 comparator is 22% at recall 50/50 (60-ep).** Do not compare against V1 §9's *30-ep* numbers — they understate by ~4–5 pp (E0 §6).

---

## Step 1 — Stage-A pretrain (once)

Fine-tune from the ImageNet-pretrained ResNeXt-101 backbone (`--pretrained`; weights wired in `resnet_design2/my_resnet.py`) — ~hours, not a multi-day pretrain.
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u main.py \
    --data data/GroupTestingDataset --task-num 2 --background-K 0 --GT-alg 1 \
    -a resnext101_32x8d --pretrained --epochs 20 --batch-size 32 --lr 0.001 \
    --output_dir Trained_Models/StageA_ITIT_ResNeXt101 --log-name stage_a.log
```
Sanity: Acc@1 high, AUC ~1.0, firearm recall 50/50 (as ResNet-18 Stage-A). Record the ResNeXt-101 **baseline** worst-case leakage too (60-ep pretrained-adv Stage-C on this checkpoint) — the frontier comparison needs the per-arch baseline (it will differ from ResNet-18's ~28%).

## Step 2 — Bracket the recall-preserving edge (~3 invariance runs, NOT a sweep)

A blind single λ_H can miss (too weak → leakage unchanged; too strong → recall < 50/50). ResNeXt-101 has more capacity, so its recall-preserving λ_H is likely *higher* than ResNet-18's 30 — **start λ_H ≈ 50–100** and adjust once or twice to find the **largest λ_H that still holds recall 50/50 @ FPR≤2%**:
```bash
# per candidate LAM (e.g. 50, then 100 or 30 depending on recall):
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m privacy.train_invariance \
    --stage-a-ckpt Trained_Models/StageA_ITIT_ResNeXt101/model_best.pth.tar \
    --data data/GroupTestingDataset --task-num 2 --background-K 0 --GT-alg 1 \
    -a resnext101_32x8d --hsic-lambda $LAM --hsic-extractor pretrained --hsic-num-random 2 \
    --pk-classes 8 --pk-per-class 4 --pk-firearm 4 \
    --epochs 30 --enc-lr 1e-4 --rec-lr 1e-4 --batch-size 32 -j 8 -valj 4 \
    --output_dir Trained_Models/Invariance_ResNeXt101_lam${LAM} --log-name invariance.log
```
After each run, measure **recall@FPR≤2%** on the 48k protocol (reuse the matched-FPR recall evaluator from `v2e0_step2_a1.py` — σ=0 here, so **no receiver adaptation needed**, just evaluate). Bracket: recall > 50/50 with margin → raise λ_H; recall < 50/50 → lower it. Stop at the largest λ_H holding 50/50.

**Watch the extractor-sanity diagnostic (M6)** on each run — the frozen pretrained extractor is now ResNeXt-101's layer3/4; confirm no WARN (true-HSIC above permuted baseline).

## Step 3 — Read the gate

At the recall-preserving λ_H from Step 2, worst-case leakage, **60-epoch** Stage-C, mean±sd over **≥3 seeds**:
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m privacy.eval_privacy \
    --stage-b-ckpt Trained_Models/Invariance_ResNeXt101_lam${BEST}/invariance_final.pth.tar \
    --data data/GroupTestingDataset --task-num 2 --background-K 0 --GT-alg 1 \
    -a resnext101_32x8d --stage-c-epochs 60 --batch-size 32 --adv-init pretrained -j 8 -valj 4 \
    --output_dir Trained_Models/StageC_Invariance_ResNeXt101_lam${BEST}_advpretrained
```
**Gate verdict:** is this leakage meaningfully below ~22% (the ResNet-18 60-ep recall-preserving number), beyond the ~1 pp band?

## Step 4 — Separability probe (free corroboration, basis-free)

On the Step-2 ResNeXt-101 checkpoint **and** a V1 ResNet-18 checkpoint (e.g. λ_H=30) as the entangled reference. New small script (no training): forward the frozen encoder on val features; fit (a) a linear **firearm direction** (the receiver's binary logit as a linear probe on pooled post-channel features) and (b) a linear **class subspace** (multinomial class probe). Report **principal angles** between the firearm direction and the class subspace, and **class-probe top-1 inside vs. in the orthogonal complement of the firearm subspace**. Disjoint (large angles; class recoverable only outside the firearm subspace) ⇒ dimensional separation — the capacity story. This interprets *why*; Step 3's leakage-at-recall adjudicates.

---

## What to record

New results doc `docs/superpowers/results/<date>-v2e2-capacity-resnext101.md`:
1. Setup (arch, batch size, 60-ep Stage-C, seeds) + ResNeXt-101 Stage-A sanity + **per-arch baseline** worst-case leakage.
2. The bracket (Step 2): λ_H tried, recall@2%FPR at each, the recall-preserving BEST.
3. **The gate number:** ResNeXt-101 worst-case leakage at recall 50/50 (60-ep, mean±sd ≥3 seeds) vs ResNet-18's ~22%.
4. Separability probe: principal angles + orthogonal-complement class accuracy, ResNeXt-101 vs ResNet-18.
5. **The one-sentence verdict + gate decision** (expand to full frontier, or stop and pivot to the tap-point question).

Commit + push the doc + scripts (checkpoints/logs gitignored, auto-synced). Add a Campaign E entry to `docs/EXPERIMENT-LOG.md`.

If the gate clears, the follow-up (spec §3 E2.2) is the full λ_H frontier + refresh-durability at BEST — a separate session.

---

## Done when
- [ ] ResNeXt-101 Stage-A trained + sanity + baseline 60-ep worst-case leakage.
- [ ] Recall-preserving λ_H found by bracketing (~3 runs), recall 50/50 @ FPR≤2% confirmed.
- [ ] Gate number: 60-ep worst-case leakage at recall 50/50, mean±sd ≥3 seeds.
- [ ] Separability probe (ResNeXt-101 vs ResNet-18).
- [ ] Results doc + EXPERIMENT-LOG Campaign E + gate verdict, committed + pushed.

**The one sentence that matters:** *At recall 50/50 (matched FPR, 60-ep), does 4× wider code get worst-case leakage meaningfully below ResNet-18's ~22% — or is the wall still there?*
