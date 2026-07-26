# HANDOFF — V2-E0: Does an SNR window exist? (frozen-checkpoint premise check)

> **This gates all of V2.** It is a cheap (~1 GPU-day) measurement on **frozen V1 checkpoints** — **no VIB code, no new mechanism training.** It answers the one question the entire "channel-noise privacy lever" thesis presupposes, before we invest in building VIB.

**To:** the next Claude instance on the GPU box (pod is alive with data + checkpoints).
**Branch:** `dev/invariance-privacy-hsic-v1`.
**Prereq reading:** `docs/superpowers/specs/2026-07-25-v2-capacity-noise-vib-design.md` (the V2 spec this check gates), `docs/superpowers/results/2026-07-25-invariance-hsic-resnet18-results.md` §9 (the V1 frontier), and `HANDOFF.md` (base mechanics).

---

## Why this, and why first

V2's headline hypothesis is that **channel noise is a privacy lever independent of representational capacity** — that adding noise destroys fine-grained class information faster than the coarse firearm decision, *even in ResNet-18's entangled code*. A design review showed this rests on an **untested premise** and that the original "rate/precision" justification does not bind (the ~100k-dim channel carries thousands of bits at any usable SNR — orders of magnitude more than the ~10 bits needed for class ID, so no "rate triage" occurs). If noise helps, the mechanism is **trained-margin asymmetry**, not channel capacity — and V1's entanglement wall is *evidence against* it. So before building VIB, measure the premise directly:

> **Is there an SNR window where firearm recall (at matched FPR) stays high while worst-case class leakage has already collapsed?**

- **Window exists** → the noise lever is real and independent → proceed with the revised V2 (raw-noise E1 on ResNet-18, capacity E2, combination E3).
- **No window (collapse knees coincide)** → the shared-subspace asymmetry is dead → **re-scope V2 capacity-first (ResNeXt-101), attach VIB only to the combination cell under a frozen-noise-floor policy.** A clean negative here saves the entire E1 build.

This check requires no VIB and no new invariance training — only the frozen V1 encoders, a short **receiver-only** noise adaptation, and Stage-C-style leakage evals at several SNRs.

---

## Threat model for σ>0 (pin these — they define the numbers)

State these in the results doc; they are load-bearing:
- **Single transmission** per image (no repetition). *Caveat to record, not test here:* channel-noise privacy is defeated by repetition averaging — an eavesdropper observing `n` transmissions recovers precision as `1/√n`; HSIC removal is immune. Note this as a scope limit of any noise-lever claim.
- **Eavesdropper SNR = legitimate receiver SNR** (same tap). *Caveat:* a physically closer eavesdropper has higher SNR; channel-noise privacy is geometry-contingent, HSIC privacy is representation-intrinsic. Note as scope.
- **Adversary trains on fresh noisy draws** (new channel noise each epoch = data augmentation = the strongest honest attacker). `eval_privacy.py` already draws fresh noise per forward, so this is automatic — confirm it, don't defeat it by caching.

---

## Prerequisites (verify before running)

```bash
cd <repo>/Over-The-Air-Neural-Group-Testing        # confirm actual path on this pod
git checkout dev/invariance-privacy-hsic-v1 && git pull
.venv/bin/python -m pytest tests/ -q -m "not slow and not gpu"     # expect 49 passed
# checkpoints present (restore from Drive if missing — see HANDOFF-v1-refine.md §1):
ls Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar \
   Trained_Models/Invariance_ResNet18_lam{30,100}/invariance_final.pth.tar
# dataset present:
ls data/GroupTestingDataset/1/train | wc -l        # expect 976; firearm val files resolve
```

**VALIDATE THE σ>0 UTILITY EVALUATOR FIRST (non-negotiable).** The feature-noise utility path must reproduce the σ=0 48k numbers exactly at `noise_std=None`, or a "no window" result could be an evaluator bug rather than a real null. Run the existing σ=0 utility eval on the Stage-A checkpoint and confirm it matches the V1 doc (Acc@1 ~97.8%, AUC 1.000, recall 50/50) before trusting any σ>0 utility number.

---

## Part A — the SNR-window sweep (the main measurement)

**Grid:** SNR ∈ {`None`(σ=0 anchor), 15, 10, 5, 0, −5, −10, −15} dB × checkpoint ∈ {Stage-A baseline, λ_H=30 (V1 BEST), λ_H=100 (strong-privacy)}. Start coarse; if a knee appears, bisect around it.

For each (SNR, checkpoint):

### A1 — Firearm utility at that SNR, with receiver-only noise adaptation
The frozen encoder's receiver was trained at σ=0; evaluating it under noise unadapted **understates** utility. So briefly adapt **R only**:
- **Freeze E** (`conv1..layer2`, `requires_grad=False`); train **R only** (`layer3..fc`) for a few epochs (e.g. 3–5, LR 1e-4) on firearm CE **with feature-level channel noise at this SNR** (reuse the joint-step channel path; do not update E, do not add HSIC/VIB). This isolates "can a receiver read the firearm bit through this much noise" from "was the receiver adapted."
- Then evaluate on the **48k protocol** with feature-level noise at this SNR:
  - Compute ROC over the 48,800 background + 50 firearm val; **pick the decision threshold at FPR ≤ 2%** (matching V1 BEST's operating point), and report **recall at that matched-FPR threshold**, plus **ROC-AUC** (threshold-free).
  - **Average over ≥3 noise draws** (different noise seeds); report **mean ± sd** for recall-at-matched-FPR and AUC.
- **Why matched-FPR:** V1 §9.4 showed recall is purchasable with false positives (a refresh bought recall 48→50 by tripling FPR). Un-pinned recall is not a valid utility axis. Pin FPR, then recall is comparable.

### A2 — Worst-case class leakage at that SNR
- Run `eval_privacy.py` with `--SNR <s> --adv-init pretrained` (worst-case attacker) and `--stage-b-ckpt <frozen ckpt>`, Stage-C 30 epochs (consider 60 at low SNR — noisy features slow adversary convergence; note if 30 hasn't converged). Leakage = worst-case top-1 over the full 48,800 background val.
- **Average over ≥3 noise draws** for the *evaluation* pass (fresh noise seeds); report mean ± sd. (Adversary *training* already sees fresh noise per epoch — confirm.)

### A3 — Read the window
For each checkpoint, plot two curves vs SNR: **recall-at-matched-FPR** (utility) and **worst-case leakage** (privacy). A **window exists** iff there is an SNR range where utility stays near its σ=0 value while leakage has dropped substantially below its σ=0 value — i.e. the two **collapse knees are separated**. If both collapse at ~the same SNR, there is **no window**.

The most informative single view: for the **Stage-A baseline** checkpoint (pure channel effect, no HSIC), overlay utility-vs-SNR and leakage-vs-SNR. This is the cleanest test of "does raw channel noise alone separate firearm from class."

---

## Part B — margin-ratio probe (static, cheap, one evening — cross-checks Part A)

On **frozen Stage-A** clean features (no noise), quantify the robustness asymmetry directly:
- **Firearm margin:** for each val image, the receiver's signed logit gap `z_firearm − z_background`. Distribution over the 48,850 val images.
- **Class margin:** for each background val image, a trained **worst-case (pretrained-adv) adversary's** top-1-vs-runner-up class logit gap. (Reuse the σ=0 worst-case adversary from the V1 baseline Stage-C run — `Trained_Models/StageC_OnStageA_ResNet18_advpretrained/` — as the class probe; if only its metrics were saved, retrain a σ=0 pretrained-adv once and keep it.)
- Convert each margin to **units of feature-space noise std**: the additive channel noise has per-element std `σ_n(SNR)`; a decision flips when noise crosses the margin in the relevant projection. Report, per SNR, the fraction of firearm decisions vs class decisions that noise would flip (or the SNR at which each distribution's median margin is crossed — the "collapse SNR" for each task).
- **Interpretation:** if the firearm collapse-SNR is well below the class collapse-SNR (firearm more noise-robust), the asymmetry supports a window and should agree with Part A. If the collapse-SNRs coincide, expect Part A to show no window. Note this is a **linear/margin approximation**; Part A (which retrains the adversary through the noise) is the arbiter — Part B is the fast sanity check.

---

## What to record

New results doc: `docs/superpowers/results/<date>-v2e0-snr-window-resnet18.md`. Include:
1. **Setup + threat model** (the three pinned choices above), and the **evaluator-validation** confirmation (σ>0 path reproduces σ=0 at `noise_std=None`).
2. **Part A tables**, per checkpoint: SNR × {recall@FPR≤2% (mean±sd), AUC (mean±sd), worst-case leakage (mean±sd)}.
3. **The two overlay plots** (utility-vs-SNR, leakage-vs-SNR) for at least the Stage-A baseline — the window verdict.
4. **Part B** margin distributions + per-task collapse-SNRs, and whether it agrees with Part A.
5. **The verdict, in one sentence:** does an SNR window exist where firearm survives and worst-case class recovery collapses? With the SNR range and the utility/leakage numbers at the window's operating point if it exists.
6. **The V2-scope implication** (see decision below).

Commit + push the results doc + any new scripts (checkpoints/logs are gitignored, auto-synced to Drive). Use/extend `parse_results.py`.

---

## The decision this produces

- **Window exists** (utility high while worst-case leakage collapsed over some SNR range): the raw-channel-noise lever is real and independent of capacity. → Revised V2 proceeds: E1 (ResNet-18 σ>0, raw noise ± HSIC), E2 (ResNeXt-101 capacity), E3 (combination + VIB). Report the window width — it sets the E1 SNR sweep.
- **No window** (collapse knees coincide — the V1-wall prediction): raw noise cannot separate firearm from class in this code. → Re-scope V2 **capacity-first (E2/ResNeXt-101)**; attach **VIB only to the combination cell (E3)** and only under a **frozen-noise-floor** policy (the only regime where VIB's power-shedding can bury below a fixed floor — under the held-SNR/re-measure policy VIB is scale-invariant and inert). Do **not** build the ResNet-18 σ>0 (E1) arm.

Either outcome is decisive and worth the ~1 GPU-day. A null is a real result — it tells us the noise thesis is dead on this arch and redirects the whole of V2.

---

## Done when

- [ ] σ>0 utility evaluator validated (reproduces σ=0 48k at `noise_std=None`).
- [ ] Part A: SNR sweep × {Stage-A, λ_H=30, λ_H=100}, R-only-adapted utility (recall@matched-FPR, AUC, mean±sd over ≥3 noise draws) + worst-case leakage (mean±sd over ≥3 draws).
- [ ] Part B: margin-ratio probe with per-task collapse-SNRs.
- [ ] Results doc with the window verdict + V2-scope implication, committed + pushed.

**The one sentence that matters:** *Is there an SNR where the firearm detector still works but the worst-case class probe has gone blind — or do they die together?*
