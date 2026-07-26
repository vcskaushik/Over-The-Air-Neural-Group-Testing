# V2 Design Spec — Two Independent Frontier Levers: Capacity and Channel-Noise/VIB

**Date:** 2026-07-25
**Status:** Design approved in brainstorming; implementation plan to follow.
**Branch (base):** `dev/invariance-privacy-hsic-v1`
**Predecessor spec:** `docs/superpowers/specs/2026-07-23-invariance-privacy-hsic-vib-design.md` (V1: HSIC, ITIT, σ=0)
**Motivating results:** `docs/superpowers/results/2026-07-25-invariance-hsic-resnet18-results.md` (§9) and `docs/EXPERIMENT-LOG.md`

---

## 1. Motivation — what V1 established, and the two levers it leaves open

V1 (HSIC invariance, ResNet-18, ITIT, σ=0) produced a **smooth, elbow-free "entanglement-wall" frontier**: against the honest worst-case (pretrained-adversary) attacker, worst-case leakage and firearm recall trade off ~proportionally (~3–5 pp leakage per missed firearm). The recall-preserving BEST (λ_H=30) cuts worst-case leakage only 27.8% → 18.6% (1.5×) at recall 48/50, and a routine 2-epoch utility refresh rebounds it to 22.3% (≈1.25× baseline). Large *and* durable privacy exists only where recall craters (λ_H=100 → 38/50).

The V1 wall is a property of **one mechanism** — HSIC — which removes privacy by **dimensional separation**: it can null class-correlated structure cheaply only if class and firearm live in *different* feature dimensions. On ResNet-18's narrow 128-channel layer-2 code they don't, so HSIC damages both in proportion.

That leaves **two independent levers untested**, each attacking the wall by a *different* mechanism:

- **Capacity (dimensional).** A wider transmitted code (ResNeXt-101: 512-ch layer-2, 4× ResNet-18) may give HSIC the room to separate class from firearm into distinct dimensions — bending the frontier.
- **Channel noise + VIB (rate / precision).** A noisy channel caps the *precision* that survives transmission. Firearm detection is a **1-bit, coarse, large-margin** decision; 976-way class ID needs **fine, high-precision** resolution. Uniform channel noise crosses the fine class margins before the coarse firearm margin, so it destroys class information *faster than* firearm information **even in a shared (entangled) subspace.** VIB (a rate penalty) actively concentrates the surviving bits on the firearm task.

**These levers are independent.** The entanglement wall is an HSIC-specific *dimensional* result; it says nothing about whether the *rate/precision* lever works, because noise exploits the coarse-vs-fine asymmetry that σ=0 never exercised. Therefore V2 does **not** gate one on the other — it characterizes both, and their combination.

**Durability note.** HSIC *hides* class structure in the encoder weights, so a firearm-only refresh can re-expose it (V1's fragility). Channel noise *destroys* fine class precision **in the channel**, which the eavesdropper never receives — structurally more likely to survive a refresh. Whether it actually does is an explicit V2 measurement, not an assumption.

---

## 2. The experiment grid

The design is the 2×2 `{ResNet-18, ResNeXt-101} × {σ=0 HSIC, σ>0 HSIC+VIB}`. The σ=0/ResNet-18 cell is V1 (done; the reference frontier). V2 adds the three new cells, **noise-first**:

| # | Cell | Lever isolated | Priority | Needs backbone pretrain? |
|---|---|---|---|---|
| **E1** | ResNet-18, **σ>0**, HSIC+VIB | **Noise/precision** (vs the rich V1 σ=0 baseline) | **First** (cheap, direct test of "does noise bend the wall HSIC couldn't?") | No (reuse V1 Stage-A) |
| **E2** | ResNeXt-101, σ=0, HSIC | **Capacity/dimensional** | Second | Use ImageNet-pretrained ResNeXt-101 (wired) |
| **E3** | ResNeXt-101, **σ>0**, HSIC+VIB | **Combination** (separate + bury) | Third | Reuse E2 Stage-A |

Everything else is held identical to V1 §9 so the frontiers overlay directly: ITIT (`--GT-alg 1 --background-K 0`), the PK sampler, and the Stage-C protocol (pretrained-adv worst-case is the primary attacker; Kaiming optional/for continuity).

---

## 3. Mechanisms and channel model

### 3.1 Combined training loss
Joint E+R training (no frozen receiver, no in-loop adversary), warm-started from the cell's Stage-A checkpoint:
```
L = L_task(E,R)  +  λ_H · HSIC(z̃, c)  +  β · KL_rate(z̃ | x)
```
- `L_task` — firearm CE, on all samples.
- `HSIC` — the V1 penalty (frozen pretrained + random-conv extractors, background-only, mean over extractors×bandwidths). Unchanged.
- `KL_rate` — the VIB rate term (below), on all samples, active only when σ>0.
All three are independently ablatable: `--hsic-lambda 0` disables HSIC; `--vib-beta 0` disables VIB; σ=0 makes `KL_rate` degenerate (guarded).

### 3.2 VIB rate term (`privacy/vib.py`)
The channel is the stochastic encoder: `z̃ = E(x) + n`, `n ~ N(0, σ²I)`. So `q(z̃|x) = N(μ(x), σ²I)` with `μ(x) = E(x)` and `σ` the channel noise std. With a fixed isotropic prior `p(z̃) = N(0, s²I)`:
```
KL_rate = E_x[ KL( N(μ(x), σ²I) ‖ N(0, s²I) ) ]
        = E_x[ Σ_d ( (σ² + μ_d(x)²)/(2 s²) + log(s/σ) − ½ ) ]   (mean over batch)
```
- **Prior scale:** `s² = coded_pwr` (auto-scaled to the signal power) by default; documented alternative `--prior-std` fixed value.
- **Guard:** raise if `KL_rate` is requested with σ = None (σ=0). Only reachable when a target SNR is set.
- This is a genuine upper bound on `I(x; z̃)`; combined with `L_task` it drives the encoder to spend transmitted rate on the firearm task and shed the rest — which is disproportionately class-carrying (the firearm task needs ~1 bit).

### 3.3 Operating point — hold the target SNR (M7 resolution)
The deployed system operates at a **target SNR**; utility and leakage are both measured there. The channel already computes `power_x`; V2 sets the noise level from the **current** transmitted power each step so the SNR is pinned to the target throughout training / Stage C / utility eval:
```
noise_std = snr_to_noise_std(target_SNR, gt_alg, coded_pwr = measured_power_x, feat_channels = <arch layer2 width>)
```
(`feat_channels` uses the arch-derived code-rate fix already landed in `_snr.py`/`main.py`.) The precision/rate mechanism works at a held SNR — the channel caps surviving precision at that SNR regardless of layout — so **power-normalization is not required**. It remains an **optional knob** (`--power-normalize`, off by default) to sharpen *per-dimension* burial if E3 warrants it; keeping it off reproduces V1/§9 exactly.

**Why "hold target SNR" is honest here (not the V1 spec's freeze):** the V1 spec §10 M7 debated freeze-vs-re-measure for a *fixed-noise-floor erasure* framing. V2's mechanism is precision-limiting at the operating point, which is well-defined by re-measuring to the target SNR. Freeze is deferred (it only matters for the sharper per-dim burial variant).

---

## 4. Sweeps (disciplined, coarse→fine like V1)

Three knobs exist (`λ_H`, `β`, SNR); to avoid a combinatorial blow-up each experiment sweeps a **primary axis** with the others pinned, then refines.

### E1 — ResNet-18, σ>0 (noise/precision lever)
Reuses V1 Stage-A. Sub-sweeps, in order:
- **E1a — channel-only frontier (λ_H=0, β=0), sweep target SNR.** Train E+R noise-aware at each SNR; the utility-vs-worst-case-leakage curve is the "free channel privacy" frontier. Also cheaply: evaluate the *frozen* V1 Stage-A encoder at each SNR (no retrain) as the zero-effort baseline. Broad SNR scan (e.g. {15, 10, 5, 0, −5, −10} dB) → refine near the utility knee.
- **E1b — HSIC + noise (β=0), sweep λ_H at the promising SNR(s) from E1a.** Does noise *complement* HSIC (super-additive) or just move along the same wall?
- **E1c — add VIB (β>0) at the best (SNR, λ_H) from E1a/b.** Does the trainable rate penalty add over raw channel noise?

### E2 — ResNeXt-101, σ=0 (capacity lever)
Mirror V1 §9 on the new arch: Stage-A from ImageNet-pretrained ResNeXt-101 → coarse λ_H {0,1,10,100,1000} → refine the recall-preserving knee → pretrained-adv Stage-C for **every** point.

### E3 — ResNeXt-101, σ>0 (combination)
At E2's recall-preserving BEST λ_H and E1's best SNR, add VIB (β sweep). Tests whether capacity-separation + noise-burial compound into privacy that is **both large and durable**.

---

## 5. Metrics, diagnostics, and the frontier comparison

Reuse the V1 §9 metric set:
- **Utility (48k):** Firearm Acc@1, ROC-AUC, recall (x/50), false-positive count + FPR. Read AUC + recall + FPR together (Acc@1 alone is the imbalanced-accuracy trap).
- **Privacy:** **pretrained-adversary (worst-case) top-1 leakage over the full background val (48,800 imgs)** — the primary number. Kaiming optional (continuity/ordering only).
- **Refresh durability:** 2-epoch utility-only refresh (E+R, LR 1e-4, firearm CE), then Stage C (worst-case) on the refreshed checkpoint. Report leakage before/after and utility (recall + FPR) before/after.

**σ>0 utility evaluator (M2).** At σ>0, utility must be evaluated with **feature-level** noise consistent with training — extend the privacy-side `validate_utility` to the full 48k protocol. Do **not** use `main.py --evaluate`, whose ITIT channel adds *pixel-space* noise (inconsistent). `eval_privacy` already taps post-channel features with feature-level noise (M1), so Stage-C leakage at σ>0 is already correct.

**Separability diagnostic (informative, not a gate).** To interpret *why* a lever does or doesn't work, add a per-channel attribution probe on a trained checkpoint: rank the transmitted channels by firearm-relevance (receiver sensitivity) and by class-relevance (adversary sensitivity); disjoint rankings ⇒ dimensional separability (favors the capacity story), overlapping ⇒ entangled (noise must rely on the precision asymmetry). This explains results; it does **not** gate any experiment.

**The headline is a frontier overlay.** Plot utility-vs-worst-case-leakage for all four cells on one axis (V1 σ=0/ResNet-18 as reference). Per lever, the question is identical: *does the frontier bend* — a larger worst-case-leakage reduction at recall ≥ ~48/50 than V1's 1.5× ceiling — *and does the operating point survive the refresh?*

---

## 6. Success criteria / what each experiment decides

- **E1 (noise lever).** If the σ>0 frontier bends past V1's σ=0 frontier at held recall (and/or is refresh-durable where V1 wasn't), the precision/rate lever is real and independent — the headline V2 result. **Null:** the frontier overlays V1's (noise is "a second knob"; firearm and class share noise-robustness). Either outcome is publishable and decisive.
- **E2 (capacity lever).** If ResNeXt-101's σ=0 frontier bends past ResNet-18's, capacity separates the tasks. **Null:** same wall at 4× width ⇒ firearm/class are dimensionally inseparable in this feature; scope the claim accordingly.
- **E3 (combination).** Whether capacity + noise compound **super-additively** into an operating point that is simultaneously (a) recall-preserving, (b) worst-case-private, and (c) **refresh-durable** — the property no single V1 point achieved.
- **Cross-cutting durability claim:** does noise-burial (E1/E3) hold under refresh where HSIC-hiding (V1) did not? This is the mechanism-level payoff.

---

## 7. Code changes

- **`privacy/vib.py`** (new) — `gaussian_kl_rate(pre_channel, noise_std, prior_std)` closed form; σ=None guard. Unit-tested.
- **`privacy/trainer.py`** — extend `joint_step` to add `β · KL_rate` (it already threads `noise_std`); background-mask stays HSIC-only, rate term on all samples.
- **`privacy/train_invariance.py`** — replace the σ=0 assert with the target-SNR path (compute `noise_std` from measured power each step); add `--vib-beta`, `--prior-std`, `--power-normalize` (default off). Keep `--arch` (E2/E3 just pass `resnext101_32x8d`).
- **`resnet_design2/my_resnet.py`** — optional power-normalization in `channel()` behind a flag, **default off** so V1/§9 reproduce byte-for-byte. Confirm `resnext101_32x8d` Stage-A load path.
- **`privacy/eval_privacy.py`** — add the 48k feature-noise **utility** evaluator (M2). Leakage path already σ>0-ready.
- **Diagnostics/scripts** — per-channel attribution probe; sweep scripts mirroring `run_hsic_refine.sh` parameterized by arch/SNR/β; `parse_results` extended for the new columns.

Backbone is already `--arch`-parametrized, so E2/E3 are mostly orchestration on top of the VIB + channel additions from E1.

---

## 8. Known risks / honest caveats

- **Three knobs → grid discipline required.** Sweep one primary axis per experiment (§4); resist full-factorial. Report what was *not* swept (no silent truncation).
- **Single seed** (as V1). Stage-C leakage has a ~1 pp run-to-run band (V1 §9.4); deltas inside it are noise. Multi-seed on the final BEST points if time allows.
- **σ>0 utility ≠ V1 utility.** At a low SNR firearm utility itself drops from channel noise alone; separate "channel cost" from "privacy mechanism cost" by always reporting the λ_H=0/β=0 noise-only baseline at each SNR (E1a).
- **Refresh re-inflation.** A firearm-only refresh could push E to transmit more rate and partially re-expose class info even under noise; this is exactly what the E1/E3 refresh tests measure — do not assume channel-destruction is automatically durable.
- **ResNeXt-101 is I/O- and compute-heavier** (86.7M params, 512-ch code); E2/E3 runs are multi-hour. Use the pretrained backbone; budget accordingly.
- **Attribution probe is a heuristic**, not proof of (in)separability; it interprets, it does not adjudicate — the frontier overlay adjudicates.

---

## 9. Out of scope (V2)

GTGT-FM / K>0 coalitions; learned VIB priors or per-element variance; CLUB; alternate tap layers; formal DP. The power-normalized per-dimension burial variant is included only as an off-by-default knob for E3, not a primary experiment.
