# HANDOFF — V2-E0: Does an SNR window exist? (frozen-checkpoint premise check)

> **STATUS: ✅ COMPLETE (2026-07-26, commit `b46b3bb`).** Results: `docs/superpowers/results/2026-07-26-v2e0-snr-window-resnet18.md`; log: `docs/EXPERIMENT-LOG.md` (Campaign D).
> **Verdict: NO usable window.** At matched recall@FPR≤2%, raw channel noise sits on/outside the σ=0 HSIC frontier (50/50: HSIC λ=30 **18.6%** vs noise −5 dB **20.0%**); the firearm/class knees coincide = the V1 entanglement wall. The static gate's ~6 dB hint was an artifact of no-adaptation + not comparing to the frontier; a Step-2 read that looked like a window was a matched-FPR error (see gotcha memory). → **Re-scope V2 capacity-first: drop E1 (noise-alone); do E2 (ResNeXt-101); VIB/noise only in E3 combination under a frozen-noise-floor** (noise+HSIC showed only a mild, unconfirmed complementarity).

> **This gates all of V2.** A **gated, cheap** measurement on **frozen V1 checkpoints** — **no VIB code, no new mechanism training.** It answers the one question the "channel-noise privacy lever" thesis presupposes, before we build VIB. **Structure: a nearly-free static GATE decides whether any expensive sweep runs at all.** Expected cost: hours (gate + a few confirm points), not a full grid.

**To:** the next Claude instance on the GPU box (pod is alive with data + checkpoints).
**Branch:** `dev/invariance-privacy-hsic-v1`.
**Prereq reading:** `docs/superpowers/specs/2026-07-25-v2-capacity-noise-vib-design.md` (the V2 spec this gates), `docs/superpowers/results/2026-07-25-invariance-hsic-resnet18-results.md` §9 (the V1 frontier), `HANDOFF.md` (base mechanics).

---

## Why this, and why first

V2's headline hypothesis is that **channel noise is a privacy lever independent of representational capacity** — adding noise destroys fine-grained class information faster than the coarse firearm decision, *even in ResNet-18's entangled code*. A design review showed the original "rate/precision" justification does **not** bind (the ~100k-dim channel carries thousands of bits at any usable SNR ≫ the ~10 bits for class ID, so no "rate triage" occurs). If noise helps at all, the mechanism is **trained-margin asymmetry** — firearm is a coarse, large-margin decision; 976-way class ID is fine and small-margin — and V1's entanglement wall is *evidence against* even that. So measure the premise directly:

> **Is there an SNR where the firearm detector still works but the worst-case class probe has gone blind — or do they die together?**

Equivalently: as you lower SNR (add noise), does **class ID collapse at a meaningfully higher SNR than firearm detection**, leaving a usable window between the two knees?

- **Window exists** → the noise lever is real and independent of capacity → build the revised V2 (E1 ResNet-18 σ>0, E2 capacity, E3 combination).
- **No window (knees coincide)** → the shared-subspace asymmetry is dead → **re-scope V2 capacity-first (ResNeXt-101)** and attach VIB only to E3 under a **frozen-noise-floor** policy. This saves the entire E1 build.

**This is deliberately gated so we do the cheapest thing that can conclude.** A single (λ_H, SNR) point cannot conclude — the window could sit at another SNR, λ_H=30 confounds the pure-noise question with HSIC, and the comparison must be against the whole σ=0 frontier. But the *full* grid is also unnecessary: the static margin probe (Step 1) locates the two collapse knees for free, and only if they look separated do we spend GPU on confirmation/characterization.

---

## Threat model for σ>0 (pin these — they define the numbers)

State in the results doc; load-bearing:
- **Single transmission** per image. *Caveat to record, not test here:* channel-noise privacy is defeated by repetition averaging (eavesdropper observing `n` transmissions recovers precision as `1/√n`); HSIC removal is immune. Scope limit of any noise-lever claim.
- **Eavesdropper SNR = legitimate receiver SNR** (same tap). *Caveat:* a closer eavesdropper has higher SNR — channel-noise privacy is geometry-contingent; HSIC privacy is representation-intrinsic. Scope limit.
- **Adversary trains on fresh noisy draws** (new channel noise each epoch = the strongest honest attacker). `eval_privacy.py` already draws fresh noise per forward — confirm it, don't defeat it by caching.

---

## Prerequisites (verify before running)

```bash
cd <repo>/Over-The-Air-Neural-Group-Testing        # confirm actual path on this pod
git checkout dev/invariance-privacy-hsic-v1 && git pull
.venv/bin/python -m pytest tests/ -q -m "not slow and not gpu"     # expect 49 passed
ls Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar \
   Trained_Models/Invariance_ResNet18_lam{30,100}/invariance_final.pth.tar   # restore from Drive if missing
ls data/GroupTestingDataset/1/train | wc -l        # expect 976; firearm val files resolve
```

**VALIDATE THE σ>0 UTILITY EVALUATOR FIRST (non-negotiable).** The feature-noise utility path must reproduce the σ=0 48k numbers at `noise_std=None`, or a "no window" could be an evaluator bug. Confirm it matches the V1 doc (Acc@1 ~97.8%, AUC 1.000, recall 50/50) before trusting any σ>0 number. (Only needed once you reach Step 2/3.)

---

## Step 1 — THE GATE: margin-ratio probe (static, nearly free, no training)

Compute where each task's decision collapses under noise, directly from **frozen** clean features — this decides whether any sweep runs.

- **Firearm margins:** on the frozen Stage-A encoder, over all val images, the receiver's signed logit gap `z_firearm − z_background`. (Use the **baseline / λ_H=0 = Stage-A** encoder — the clean pure-noise question; λ_H=30/100 have HSIC applied and are for Step 3 only.)
- **Class margins:** over the 48,800 background val images, a trained **worst-case (pretrained-adv) adversary's** top-1-vs-runner-up class logit gap. Reuse the σ=0 worst-case adversary from the V1 baseline Stage-C run (`Trained_Models/StageC_OnStageA_ResNet18_advpretrained/`); if only metrics were saved, retrain one σ=0 pretrained-adv and keep the weights.
- **Convert to a collapse-SNR per task.** The channel adds per-element noise of std `σ_n(SNR)` (use `privacy/_snr.py`, ITIT code-rate=1.0, `coded_pwr` = measured Stage-A feature power). A decision flips when noise of that scale crosses the margin in the decision's projection. For each task report the **collapse-SNR** — the SNR at which the median margin equals the effective noise std (and, better, the full curve: fraction of decisions flipped vs SNR).
- **GATE RULE.** As SNR drops, the smaller-margin task fails first (at *higher* SNR).
  - **Class-collapse-SNR meaningfully ABOVE firearm-collapse-SNR** (class dies first, firearm survives to lower SNR) → a window plausibly exists in that band → **go to Step 2 to confirm.**
  - **Collapse-SNRs coincide (or firearm collapses first)** → no window is possible at any SNR → **go straight to Step 2 for a 2–3-point confirmation, then STOP** (no full grid).

This is a **linear/margin approximation** — Step 2 (which retrains the adversary through the noise) is the arbiter; Step 1 just tells you whether it's worth running and where to point it.

---

## Step 2 — CONFIRM on the baseline (minimal: ~3 SNR points, Stage-A only)

Confirm the gate's prediction with real retrained-adversary numbers on the **baseline (Stage-A / λ_H=0)** encoder — the clean pure-noise test. Pick ~3 SNRs bracketing the gate's predicted knee(s) — e.g. **{5, 0, −5} dB** if the gate is ambiguous, or one SNR just inside and one just outside a predicted window.

For each SNR:
- **A1 — Firearm utility (receiver-only noise adaptation).** Freeze E (`conv1..layer2`); train **R only** (`layer3..fc`) a few epochs (3–5, LR 1e-4) on firearm CE **with feature-level noise at this SNR** (reuse the joint-step channel path; no E update, no HSIC/VIB). Then on the 48k protocol at this SNR: compute ROC, pick the threshold at **FPR ≤ 2%** (V1 BEST's operating point), report **recall at that matched-FPR threshold** + **AUC**, each **averaged over ≥3 noise draws (mean ± sd)**. *(Matched-FPR because V1 §9.4 showed recall is purchasable with false positives — un-pinned recall is not a valid utility axis.)*
- **A2 — Worst-case class leakage.** `eval_privacy.py --SNR <s> --adv-init pretrained --stage-b-ckpt Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar`, Stage-C 30 epochs (60 at low SNR if unconverged). Leakage = worst-case top-1 over the 48,800 background val, **averaged over ≥3 eval noise draws (mean ± sd)**.

**Read:** compare each point to the σ=0 **frontier** (V1 §9), at *matched utility*. The noise point must land **inside** the frontier (lower leakage at equal recall-at-FPR) to count as "noise helps." If the confirm points reproduce the gate's "knees coincide" → **conclude no window, STOP, re-scope V2 (see decision).** If they confirm a separation → Step 3.

---

## Step 3 — CONDITIONAL: characterize the window (only if Steps 1–2 show one)

Only if a window looks real, quantify it (needed to scope E1's SNR sweep). Expand Step 2 to the fuller SNR set {15,10,5,0,−5,−10,−15} dB, and add the **λ_H=30 and λ_H=100** checkpoints (does noise complement HSIC, or just move along the same wall?). Bisect around the knee. Same A1/A2 protocol. This is the *characterization* grid — do **not** run it for the go/no-go.

---

## What to record

New results doc `docs/superpowers/results/<date>-v2e0-snr-window-resnet18.md`:
1. **Setup + threat model** (the three pinned choices) + **evaluator-validation** confirmation.
2. **Step 1 (gate):** firearm vs class margin distributions and per-task collapse-SNRs; the gate verdict (separated / coincident).
3. **Step 2 (confirm):** the ~3 baseline points — SNR × {recall@FPR≤2%, AUC, worst-case leakage} (all mean ± sd) — and whether they land inside or on the σ=0 frontier.
4. **Step 3 (if run):** the characterization grid + the two overlay curves (utility-vs-SNR, leakage-vs-SNR).
5. **The verdict, one sentence:** does an SNR window exist? With the band and the operating-point utility/leakage if so.
6. **The V2-scope implication** (below).

Commit + push the results doc + any new scripts (checkpoints/logs gitignored, auto-synced). Use/extend `parse_results.py`.

---

## The decision this produces

- **Window exists** (Step 2 baseline points land inside the σ=0 frontier over some SNR band): raw channel noise separates firearm from class independent of capacity. → Revised V2: E1 (ResNet-18 σ>0, raw noise ± HSIC), E2 (ResNeXt-101 capacity), E3 (combination + VIB); the window band sets the E1 SNR sweep.
- **No window** (knees coincide / points sit on the frontier — the V1-wall prediction): raw noise cannot separate them in this code. → Re-scope V2 **capacity-first (E2/ResNeXt-101)**; attach **VIB only to E3** and only under a **frozen-noise-floor** policy (under the held-SNR/re-measure policy VIB is scale-invariant and inert). Do **not** build the ResNet-18 σ>0 (E1) arm.

A null is a real, decisive result — it kills the noise thesis on this arch and redirects V2. That is exactly why the gate is cheap: we spend the minimum that can reach it.

---

## Done when

- [x] σ>0 utility evaluator validated (reproduces σ=0 48k at `noise_std=None`: AUC 0.9998, recall 50/50).
- [x] **Step 1 gate** run: class ≈ −4 dB vs firearm ≈ −10 dB half-signal (looked separated — but static, non-arbiter).
- [x] **Step 2 confirm:** baseline {0,−5,−10} dB, R-adapted recall@2%FPR + worst-case leakage (adv retrained through noise), mean±sd.
- [x] **Step 3** matched-FPR frontier overlay + noise+HSIC combination — the arbiter; overturned the Step-2 hint.
- [x] Results doc with the (null) window verdict + V2-scope implication, committed + pushed (`b46b3bb`).

**The one sentence that matters:** *Is there an SNR where the firearm detector still works but the worst-case class probe has gone blind — or do they die together?*
