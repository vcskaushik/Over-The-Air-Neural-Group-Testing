# V2-E0 results: does an SNR window exist? (ResNet-18, ITIT, frozen V1 checkpoints)

**Date:** 2026-07-26 · **Branch:** `dev/invariance-privacy-hsic-v1` · **Gates:** `docs/superpowers/specs/2026-07-25-v2-capacity-noise-vib-design.md`
**Premise checked:** is channel noise a privacy lever *independent* of representational capacity — is there an SNR where the firearm detector still works but the worst-case class probe has gone blind, that *beats* the σ=0 HSIC frontier?

> **Headline: NO usable window — raw channel noise does NOT beat HSIC.** Measured at **matched recall @ FPR≤2%**, adding channel noise to the baseline encoder sits **on/just outside** the σ=0 HSIC frontier, not inside it. At the recall-preserving point (50/50 @ 2%FPR), **HSIC λ_H=30 leaks 18.6% while the best pure-noise point (−5 dB) leaks 20.0%** — noise is *marginally worse*. The two mechanisms' collapse knees effectively **coincide** — exactly the V1 entanglement-wall prediction. Channel noise is therefore **not** an independent lever here → **re-scope V2 capacity-first (E2/ResNeXt-101); attach VIB/noise only in E3 (combination), and drop the noise-alone E1 arm.** (One nuance: noise+HSIC *combined* shows a mild, unconfirmed complementarity — see §4.)

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
