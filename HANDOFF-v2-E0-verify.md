# HANDOFF — V2-E0 verification: close the null at the letter level

> **A cheap (~few GPU-hours) follow-up to Campaign D (V2-E0).** An independent audit confirmed the **decision** — V2 goes capacity-first, drop the standalone-noise E1 arm — is sound and safe. But it found the E0 write-up **overreaches in three fixable places**. This handoff runs the minimal checks to either close the null at the letter level or honestly document a small (threat-model-fragile) window, and corrects the results doc.

**To:** the next Claude instance on the GPU box (frozen V1 checkpoints; no new mechanism, no VIB).
**Branch:** `dev/invariance-privacy-hsic-v1`.
**Prereq reading:** `docs/superpowers/results/2026-07-26-v2e0-snr-window-resnet18.md` (the doc to correct), `HANDOFF-v2-E0-snr-window.md` (the original gated design), `docs/superpowers/results/2026-07-25-invariance-hsic-resnet18-results.md` §9 (the V1 σ=0 HSIC frontier).

## Why (what the audit found)

The audit recomputed every E0 number from the JSONs and confirmed: the matched-FPR comparison is apples-to-apples, the two evaluator bugs did **not** corrupt any E0 number (main.py's pixel-noise path is never called by the E0 scripts; the `device.index` bug *crashes*, it doesn't silently drop noise), and noise was applied at the claimed magnitude. **The thesis ("an SNR where firearm works but the class probe is blind") is decisively answered no** — at every SNR with recall ≥ 0.93, worst-case leakage ≥ 11.2% (~110× chance). Capacity-first is safe **regardless of what this handoff finds** — a small window here would be ≤ ~3 pp and defeated by repetition-averaging / a closer eavesdropper (threat-model fragility), not worth an E1 arm.

But three claims in the doc outrun the data:
1. **"No window at 50/50 recall" rests on an unprobed SNR band.** We have recall 1.0 @ −5 dB (leak 20.0%) and 0.933 @ −10 dB (leak 11.2%). **−7 and −8 dB were never run.** If recall stays 50/50 there with leakage < ~17%, a genuine recall-preserving window exists.
2. **The §4 "noise on/outside the frontier" claim self-contradicts at ~47/50**, and it compares noise's matched-FPR recall against HSIC λ50's *argmax* recall — the exact mismatch the doc criticizes. Matched-FPR recall was re-measured only for λ0/30/100, never λ20/50/70.
3. **The decisive 18.6% (HSIC λ30) vs 20.0% (noise −5 dB) gap is inside the ~1 pp cross-campaign noise floor** (E0's own σ=0 baseline rerun read 26.7% vs V1's 27.80% — a −1.1 pp offset). It's a **tie on the same wall**, not "HSIC marginally wins."

And E3's design premise — a "mild noise+HSIC complementarity" — is a 1.5 pp single-run edge inside a ~1 pp band. Unconfirmed.

## What to run (reuse the existing E0 scripts)

All on **frozen** checkpoints; reuse `v2e0_step2_a1.py` (A1: receiver-adapted recall@2%FPR), `v2e0_step2_a2.py` (A2: worst-case Stage-C leakage), `v2e0_step3.py`, `v2e0_train_adv.py`. Baseline = `Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar`.

### V1 — Fill the −5→−10 dB gap (the one check that could reveal a window)
On the **baseline** encoder, at **SNR ∈ {−7, −8} dB**:
- **A1 (utility):** receiver-only noise adaptation → recall@FPR≤2% + AUC, mean±sd over ≥3 noise draws. **Double the adaptation budget** (10 epochs instead of 5, LR 1e-4, R-only) — the audit flagged 5 epochs may understate low-SNR recall and *mask* a window.
- **A2 (leakage):** worst-case pretrained-adv Stage-C, 30 epochs, mean±sd over ≥3 eval noise draws.
- **Window criterion:** recall ≥ ~0.98 (≈49–50/50) **and** worst-case leakage ≤ ~17% (meaningfully below HSIC λ30's 18.6% at matched recall, beyond the ~1 pp band). If met → a small window exists (document it, note threat-model fragility). If recall < 0.98 or leakage ≥ ~17% → the null is closed at the letter level.

### V2 — Matched-FPR recall for λ20/50/70 (near-free, no training)
Run only `eval_recall`@2%FPR (the A1 evaluator, **no receiver adaptation needed** — these are σ=0 HSIC checkpoints already jointly trained) on `Trained_Models/Invariance_ResNet18_lam{20,50,70}/invariance_final.pth.tar`. This replaces the doc's argmax-recall references so the noise-vs-HSIC frontier comparison at ~47/50 is matched-FPR on both sides. Minutes each.

### V3 — Confirm (or kill) the E3 complementarity hint
Repeat the combo point **λ30 + −5 dB** (`v2e0_step3.py` path) with a **second seed** (different noise + init seed). Report both seeds' worst-case leakage at matched recall. If the ~1.5 pp edge over pure noise survives both seeds → E3's premise is real; if it collapses into the band → E3 should not lean on it.

### V4 (optional, if time) — convergence check
60-epoch Stage-C at −5 and −10 dB on the baseline. The audit noted an adversary trained *through noise* may converge slower at 30 epochs, which would **understate** noise leakage (flattering noise). If 60-epoch leakage rises materially, the null is even safer; if unchanged, 30 epochs was fine.

## What to record — correct the existing results doc

**Edit `docs/superpowers/results/2026-07-26-v2e0-snr-window-resnet18.md`** (don't start a new file); add a "Verification / corrections" section and fix the three overreaches:
1. Add the −7/−8 dB rows; state whether a recall-preserving window exists and, if so, that it is ≤ ~3 pp and threat-model-fragile (repetition-averaging + eavesdropper-SNR), hence does **not** revive E1.
2. Replace the §4 λ20/50/70 argmax-recall references with the matched-FPR recalls (V2); re-state whether noise sits inside or on the frontier at ~47/50 with both sides matched.
3. Soften "noise is marginally worse than HSIC" → **"tied on the same wall within the ~1 pp cross-campaign Stage-C band"** (cite the 26.7% vs 27.80% σ=0 baseline offset).
4. One sentence on the threshold-on-val caveat: the 2%-FPR threshold is picked (oracle-recalibrated per noise draw) on the same val it reports; identical for both arms so verdict-neutral, but a deployed receiver couldn't recalibrate per-realization.
5. State the E3 complementarity result (V3): confirmed or noise-floor.

Keep the **overall verdict unchanged**: no usable window; V2 capacity-first; standalone-noise E1 dropped. Commit + push the corrected doc + any new numbers (checkpoints/logs gitignored, auto-synced).

## Done when
- [x] −7, −8 dB: A1 (10-ep-adapted recall@2%FPR + AUC) and A2 (worst-case leakage), mean±sd over ≥3 draws. (−7: 0.993 @ 16.73%→**21.77% @60ep**; −8: 0.973 @ 14.96%.)
- [x] λ20/50/70 matched-FPR recall re-measured. (0.98 / 0.94 / 0.90.)
- [x] Combo λ30+−5 dB repeated on a 2nd seed. (seed0 9.66%, seed1 9.75% — seed-robust but 30-ep under-converged → not established.)
- [x] 60-epoch Stage-C — ran at −5/−10 dB **and** the clinching HSIC λ30 / baseline −7 dB points. **Finding: BOTH arms under-converge at 30 ep (+3.5–5.4 pp).**
- [x] Results doc corrected (§6): 3 overreaches + E3 status; verdict unchanged (no window; tie on same wall); committed + pushed.

> **RESULT:** No usable window. At converged (60-ep) matched-FPR, HSIC λ=30 (22.14%) ≈ noise −7 dB (21.77%) — a tie on the same wall. The 30-ep "window/complementarity" hints were convergence artifacts. V2 stays capacity-first; E1 dropped.

**The one sentence that matters:** *Does a recall-preserving (≥49/50) window with leakage < ~17% appear at −7/−8 dB — and does the noise+HSIC combo edge survive a second seed?* Either way, V2 stays capacity-first; this just makes the record honest and grounds E3.
