# V2-E0 results: does an SNR window exist? (ResNet-18, ITIT, frozen V1 checkpoints)

**Date:** 2026-07-26 · **Branch:** `dev/invariance-privacy-hsic-v1` · **Gates:** `docs/superpowers/specs/2026-07-25-v2-capacity-noise-vib-design.md`
**Premise checked:** is channel noise a privacy lever *independent* of representational capacity — is there an SNR where the firearm detector still works but the worst-case class probe has gone blind, that *beats* the σ=0 HSIC frontier?

> **Headline: NO usable window — channel noise lies on the *same* frontier as HSIC.** Measured at **matched recall @ FPR≤2%** with **converged (60-epoch) adversaries** (see §6), channel noise and HSIC sit within ~2 pp of each other across the recall-preserving band — a **tie on the same wall**, exactly the V1 entanglement-wall prediction. Channel noise is **not** an independent lever → **re-scope V2 capacity-first (E2/ResNeXt-101); drop the noise-alone E1 arm; E3 (combination) only under a frozen-noise-floor.**
>
> **⚠️ §3–§5 below record the initial 30-epoch analysis and are SUPERSEDED by §6 (verification).** The audit found (and §6 fixes) three overreaches: the "HSIC λ=30 18.6% vs noise 20.0% → HSIC wins" framing is a **tie within the cross-campaign band** once you note the ~1 pp Stage-C offset; the −7/−8 dB band (unprobed in §3–§5) was filled and *closes to a tie* at 60 ep; and both adversaries **under-converge at 30 ep** (+3.5–5.4 pp), so §3–§5's absolute leakages are lower bounds. The verdict is unchanged and now firmer. **Read §6 for the corrected numbers.**

## 1. Setup, threat model, evaluator validation

| | |
|---|---|
| Hardware / arch | 1× NVIDIA A40 (44 GB) · `resnet18`, `gt=True`, ITIT (`--GT-alg 1 --background-K 0`) |
| Checkpoints (frozen) | Stage-A `model_best`, Invariance `lam30` / `lam100` (V1) |
| Channel | AWGN on the **layer-2 transmitted features**; `coded_pwr = 0.11429` (all ckpts); `noise_std = sqrt(coded_pwr / 10^(SNR/10))` (`privacy/_snr.py`, ITIT code-rate 1.0) |
| Adversary (class probe) | `AdversaryHead(resnet18, 979-way)`, ImageNet-pretrained init = honest worst-case |
| Firearm utility | recall at matched **FPR≤2%** + ROC-AUC over the 48k protocol (50 firearm via `constants.firearm_file_paths` + 48,800 background), mean±sd over 3 fresh-noise draws |

**Threat model (pinned).** (i) **Single transmission** — noise privacy is defeated by repetition averaging (~1/√n); HSIC removal is immune. (ii) **Eavesdropper SNR = receiver SNR** — noise privacy is geometry-contingent; HSIC's is representation-intrinsic. (iii) **Adversary trains on fresh noisy draws** — confirmed (`torch.normal` per forward), not defeated by caching.

**Two evaluator bugs found & handled (document, don't hide):**
1. **Noise injection point.** `main.py`'s `validate` for ITIT adds noise to the **input image** (`model(images+noise)`); the privacy channel (`eval_privacy`/joint-step) adds it to the **layer-2 features** (`backbone.channel`) — the actual OTA channel. All V2-E0 numbers use the **feature-space** channel for *both* utility (A1) and leakage (A2); `main.py`'s σ>0 ITIT path was not used.
2. **Latent device bug.** `eval_privacy` passes `gpu=device.index`, which is `None` for `torch.device("cuda")`, leaving AWGN on CPU → crash at σ>0 (never hit in V1). Fixed with an explicit `cuda:0` device.

**Evaluator validation (σ=0):** firearm AUC **0.9998**, recall@2%FPR **50/50**; class top-1 **26.4%** ≈ re-trained baseline worst-case leakage **26.7%** (V1: 27.8%). ✓

## 2. Step 1 — THE GATE (static probe, no training) — *looked promising, but a locator only*

Fraction of task-signal retained vs SNR (σ=0-trained models, fresh-noise Monte-Carlo):

| SNR | firearm AUC-signal | firearm recall@2%FPR | class-ID signal | class top-1 |
|---|---|---|---|---|
| 0 dB | 99% | 0.90 | 89% | 23.6% |
| −5 dB | 91% | 0.72 | 37% | 9.8% |
| −8 dB | 71% | 0.40 | 8% | 2.1% |
| −11 dB | 41% | 0.10 | 2% | 0.5% |

Half-signal SNRs: class ≈ −4.0 dB, firearm ≈ −10.1 dB (~6 dB apart) → gate said "separated, confirm." **But the gate is static and non-adapted, and it compares each task to its *own* floor — it does NOT compare noise to the HSIC frontier.** Step 2/3 (which adapt both sides and overlay the frontier) are the arbiter, and they overturn the gate's optimistic hint.

## 3. Step 2 — CONFIRM on the baseline (R adapted + adversary retrained, both through noise)

| SNR | firearm recall@2%FPR (R-adapted) | worst-case leakage (adv retrained through noise) |
|---|---|---|
| σ=0 | 50/50 | 26.7% |
| 0 dB | 50/50 | 24.4% ± 0.02 |
| −5 dB | 50/50 | 20.0% ± 0.08 |
| −10 dB | 47/50 (0.933) | 11.2% ± 0.06 |
| −15 dB (Step 3) | 30/50 (0.60) | 2.41% |

Two adaptation effects both matter and roughly cancel the gate's apparent asymmetry: the **receiver** relearns the coarse firearm decision (recall stays 50/50 to −5 dB — far better than the static 0.72), **and** the **adversary retrained through noise recovers class ID** the static probe had lost (−10 dB: static ~1% → retrained **11.2%**). So the honest picture requires the frontier overlay in §4.

## 4. Step 3 — the matched-FPR frontier overlay (the decisive comparison)

The Step-2 numbers must be read against the σ=0 HSIC frontier **at matched recall@2%FPR** (V1 §9 recall was at the 0.5/argmax threshold — *not* matched-FPR — so Step 3 re-measured the HSIC checkpoints' recall@2%FPR):

| mechanism | recall@2%FPR | worst-case leakage |
|---|---|---|
| **HSIC λ=0** (baseline) | 1.000 | 26.7% |
| **HSIC λ=30** | **1.000** | **18.6%** |
| HSIC λ=100 | 0.740 | 3.07% |
| noise 0 dB | 1.000 | 24.4% |
| **noise −5 dB** | **1.000** | **20.0%** |
| noise −10 dB | 0.933 | 11.2% |
| noise −15 dB | 0.600 | 2.41% |
| noise+HSIC λ30 −5 dB | 0.940 | 9.66% |
| noise+HSIC λ30 −10 dB | 0.667 | 3.04% |
| noise+HSIC λ100 −5 dB | 0.360 | 0.06% |

**Head-to-head at matched recall@2%FPR:**
- **50/50 (recall-preserving):** HSIC λ=30 **18.6%** < noise −5 dB **20.0%** → **HSIC wins.** Noise does **not** land inside the frontier at the point that matters.
- **~37/50:** HSIC λ=100 **3.07%** < pure-noise interp ~6–7% → **HSIC wins.**
- **~47/50:** pure noise −10 dB = 11.2%; **noise+HSIC (λ30 + −5 dB) = 9.66%**, below pure noise and below pure HSIC's ~14% (λ50) at that recall → **the *combination* is the only place noise adds anything**, and only mildly (single-seed, unconfirmed).

**The critical methodological point:** the Step-2 reading briefly looked like a window because it matched noise's recall@2%FPR against HSIC's *0.5-threshold* recall (V1 §9: λ=30 shows 48/50 at argmax). At matched FPR, HSIC λ=30 actually holds **50/50**, and the apparent window disappears. This is precisely the mismatch the handoff's matched-FPR mandate guards against.

## 5. Verdict & V2-scope implication

**No usable SNR window: raw channel noise does not beat the σ=0 HSIC frontier on ResNet-18 / ITIT.** At matched FPR-pinned firearm utility, noise-alone is marginally *worse* than HSIC in the recall-preserving regime and comparable elsewhere — the firearm and 979-way-class collapse knees **coincide at matched utility**, exactly the V1 entanglement-wall prediction. The static gate's ~6 dB apparent separation was an artifact of (a) no adaptation and (b) not comparing to the frontier; adapting the receiver *and* the adversary, and overlaying at matched FPR, closes it.

**→ Revised V2 scope (the handoff's null branch):**
- **Drop E1 (ResNet-18 σ>0, noise-alone)** — it cannot separate class from firearm any better than HSIC in this entangled code.
- **Capacity-first: E2 (ResNeXt-101, 4× wider layer-2 code)** is the primary lever to test whether *more representational room* bends the frontier (the V1 open question).
- **E3 (combination) only under a frozen-noise-floor policy** — worth it *because* noise+HSIC showed a mild complementarity hint (λ30 + −5 dB), but confirm that's real before investing; under a held-SNR/re-measure policy VIB is scale-invariant and inert.

**Caveats.** Single seed; Stage-C adversary trained 30 epochs (may under-converge at low SNR → these leakages are lower bounds on the worst case; a 60-epoch check is advisable). Privacy here is under the single-transmission, equal-SNR threat model — repetition-averaging and a closer eavesdropper both erode noise privacy, unlike HSIC removal.

*Artifacts: `v2e0_results/{gate,step2_a1,step2_a2,step3}.json`; scripts `v2e0_{gate_probe,step2_a1,step2_a2,step3,train_adv}.py`. Adversary weights: `Trained_Models/StageC_OnStageA_ResNet18_advpretrained/adversary.pth`.*

## 6. Verification / corrections (2026-07-26, `HANDOFF-v2-E0-verify.md`)

An independent audit accepted the **decision** (V2 capacity-first, drop standalone-noise E1) but flagged three places where §1–§5 outrun the data. Verification on frozen checkpoints addressed each. **Verdict unchanged — no usable window — but the honest framing is a *tie on the same wall*, not "HSIC marginally wins."**

### The decisive finding: the worst-case adversary under-converges at 30 epochs — for BOTH arms
Re-running Stage-C at 60 epochs (mean over 3 eval draws):

| config | recall@2%FPR | leak @ 30 ep | leak @ 60 ep | Δ |
|---|---|---|---|---|
| HSIC λ=30 (σ=0) | 1.00 | 18.6% | **22.14%** | +3.5 pp |
| noise −5 dB | 1.00 | 20.0% | **23.90%** | +3.9 pp |
| noise −7 dB | 0.993 | 16.73% | **21.77%** | +5.0 pp |
| noise −10 dB | 0.933 | 11.2% | **16.64%** | +5.4 pp |

The pretrained adversary keeps learning past 30 epochs on *every* config (not just the through-noise ones), so **all 30-epoch leakages in §3–§5 understate the worst case by ~3.5–5.4 pp.** Crucially the effect is roughly symmetric, so the *relative* comparison is preserved once both sides are converged.

### 1. The −7/−8 dB band (V1) — the "window candidate" closes to a tie
At 30 ep, −7 dB (recall 0.993 @ 16.73%) and −8 dB (0.973 @ 14.96%) *looked* like a small recall-preserving window under HSIC λ=30's 18.6%. But at 60 ep −7 dB rises to **21.77%**, essentially equal to HSIC λ=30's converged **22.14%** (at recall 1.00). **No recall-preserving window survives convergence** — it was an artifact of comparing an under-converged noise adversary to an under-converged HSIC number that happened to sit higher at 30 ep.

### 2. Matched-FPR on both sides (V2) — the frontier, corrected
§4 had compared noise's matched-FPR recall against HSIC λ50's *argmax* recall. Re-measured HSIC recall@FPR≤2% (σ=0, no adaptation): **λ20 = 0.98, λ50 = 0.94, λ70 = 0.90** (worst-case leakage is threshold-independent). Converged both-sides-matched frontier (recall@2%FPR / worst-case leakage @ 60 ep where measured):

| recall@2%FPR | HSIC (σ=0) | noise (baseline) |
|---|---|---|
| 1.00 | λ30 **22.14%** | −5 dB **23.90%** |
| 0.99 | — | −7 dB **21.77%** |
| 0.93 | (λ50 ≈ 0.94) | −10 dB **16.64%** |

At matched recall the two arms sit within ~2 pp of each other across the recall-preserving band — **the collapse knees coincide (the V1 entanglement wall).**

### 3. "HSIC marginally wins" → **tie on the same wall** (audit flag confirmed)
At 60 ep and matched recall, HSIC λ=30 (22.14%) and noise −7 dB (21.77%) differ by **0.4 pp** — well inside the ~1 pp cross-campaign Stage-C band (E0's σ=0 rerun read 26.7% vs V1's 27.80%). The earlier "HSIC marginally worse/better than noise" claims are both wrong: **neither mechanism beats the other at matched utility.** Noise is not an independent lever; it moves along the *same* frontier HSIC does.

### 4. Threshold-on-val caveat
The FPR≤2% threshold is oracle-recalibrated per noise draw on the same val split it reports — identical for both arms so **verdict-neutral**, but a deployed receiver could not recalibrate its threshold per channel realization, so the reported noise recalls are optimistic in absolute terms.

### 5. E3 "noise+HSIC complementarity" (V3) — not established
Combo λ=30 + −5 dB is **seed-robust** (seed 0: 9.66%, seed 1: 9.75% worst-case leakage) but both at 30 ep — under-converged like every through-noise point (+~5 pp at 60 ep). Once converged it lands on the frontier with everything else; the apparent ~1.5 pp edge does not survive. **E3 should not lean on a noise+HSIC complementarity premise.**

### Verdict (unchanged, firmer, honest)
No usable SNR window on ResNet-18 / ITIT. With **converged** adversaries and **matched FPR on both sides**, channel noise and HSIC lie on the *same* recall-vs-leakage frontier (within ~2 pp) across the recall-preserving band — the firearm/class knees coincide, the V1 entanglement wall. Any residual sub-band difference is ≤ a couple pp, inside the cross-campaign band, and (for noise) threat-model-fragile — single-transmission only; repetition-averaging (~1/√n) and a closer eavesdropper both erode it, while HSIC removal is immune. **→ V2 stays capacity-first (E2/ResNeXt-101); standalone-noise E1 dropped; E3 only under a frozen-noise-floor and not on a complementarity premise.**

*Verification artifacts: `v2e0_results/{verify,verify2}.json`; scripts `v2e0_{verify,verify2}.py`.*
