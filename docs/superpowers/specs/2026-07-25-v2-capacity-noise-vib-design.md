# V2 Design Spec — Capacity-First (with Noise/VIB only in the combination cell)

**Date:** 2026-07-25 (**revised 2026-07-26** after Campaign D)
**Status:** Re-scoped capacity-first (supersedes the noise-first draft); implementation plan to follow.
**Branch (base):** `dev/invariance-privacy-hsic-v1`
**Predecessor spec:** `docs/superpowers/specs/2026-07-23-invariance-privacy-hsic-vib-design.md` (V1: HSIC, ITIT, σ=0)
**Motivating results:** `docs/superpowers/results/2026-07-25-invariance-hsic-resnet18-results.md` §9 (V1 frontier), `docs/superpowers/results/2026-07-26-v2e0-snr-window-resnet18.md` (Campaign D / E0), `docs/EXPERIMENT-LOG.md`

> **Revision note.** The original V2 draft led with a "noise-first" ResNet-18 σ>0 experiment (E1) on the hypothesis that channel noise is a privacy lever *independent* of representational capacity. **Campaign D (E0) tested that premise on frozen V1 checkpoints and returned a null:** at matched recall@FPR≤2%, raw channel noise sits on the σ=0 HSIC frontier (tied within the ~1 pp Stage-C band; 50/50 recall → HSIC λ30 18.6% vs noise −5 dB 20.0%) — the firearm/class collapse-knees coincide = the same entanglement wall. So **the standalone-noise E1 arm is dropped.** The audit-flagged loose ends (an unprobed −7/−8 dB band; matched-FPR recall for λ20/50/70; a 2nd seed for the noise+HSIC combo) are being closed by `HANDOFF-v2-E0-verify.md`; the capacity-first decision holds regardless of what that finds (any residual window is ≤ ~3 pp and threat-model-fragile).

---

## 1. Motivation — capacity is the untested lever

V1 established a smooth, elbow-free **entanglement-wall frontier** on ResNet-18/ITIT/σ=0: against the honest worst-case (pretrained-adversary) attacker, worst-case leakage and firearm recall trade off ~proportionally; the recall-preserving BEST (λ_H=30) cuts worst-case leakage only 27.8%→18.6% (1.5×), and a 2-epoch refresh rebounds it to 22.3%. The wall is a property of the HSIC mechanism, which removes privacy by **dimensional separation** and can do so cheaply only if class and firearm live in *different* dimensions — which they don't in ResNet-18's narrow 128-channel layer-2 code.

Two levers could bend it, by different mechanisms:
- **Capacity (dimensional).** A wider transmitted code (ResNeXt-101: 512-ch layer-2, 4× ResNet-18) may give HSIC the room to separate class from firearm into distinct dimensions. **Untested — now the primary V2 experiment.**
- **Channel noise / VIB (rate·precision).** *Tested in Campaign D on ResNet-18 and found null as a standalone lever* — raw noise moves along the same wall (firearm and 976-way class have similar noise-robustness in the shared subspace, so uniform noise degrades both together). It survives only as a possible **combination** knob: noise+HSIC showed a mild, unconfirmed complementarity, and *if capacity separates the tasks*, noise could bury the separated class dimensions **durably** (physical channel destruction vs HSIC's weight-space hiding, which a refresh re-exposes).

So V2 leads with capacity (E2) and keeps noise/VIB only as a conditional combination arm (E3).

---

## 2. Experiments

The relevant grid is `{ResNet-18, ResNeXt-101} × {σ=0 HSIC, σ>0 HSIC+VIB}`. Two cells are settled: **ResNet-18/σ=0** = V1 (the reference frontier); **ResNet-18/σ>0** = Campaign D (null, dropped). V2 runs the two ResNeXt-101 cells:

| # | Cell | Question | Priority |
|---|---|---|---|
| **E2** | ResNeXt-101, σ=0, HSIC | Does 4× capacity **bend the frontier** — a given worst-case-leakage reduction at *less* recall cost, durable under refresh? | **Primary** |
| **E3** | ResNeXt-101, σ>0, HSIC+VIB | Given E2's separability, does noise/VIB (under a **frozen-noise-floor**) make the recall-preserving privacy **refresh-durable**, and does the noise+HSIC complementarity confirm? | Conditional / secondary |

Everything else is held identical to V1 §9 so frontiers overlay: ITIT (`--GT-alg 1 --background-K 0`), PK sampler, and the Stage-C protocol (pretrained-adv worst-case primary).

---

## 3. E2 — Capacity sweep (primary)

- **Arch:** `resnext101_32x8d` — layer-2 code 512 ch (4× ResNet-18), ImageNet-pretrained weights (wired in `resnet_design2/my_resnet.py`) to skip a multi-day backbone pretrain. **Only** this lever changes vs V1 §9.
- **Sweep:** Stage-A pretrain → coarse λ_H {0,1,10,100,1000} → refine the recall-preserving knee → **pretrained-adv (worst-case) Stage-C for every point** (Kaiming optional). Mirror `run_hsic_refine.sh` on the new arch.
- **Primary metric — frontier comparison.** Overlay ResNeXt-101's utility-vs-worst-case-leakage frontier on ResNet-18's §9.3. *Bent* = a larger worst-case-leakage reduction at recall ≥ ~48/50 than V1's 1.5× ceiling. **Report per-arch baseline and both absolute and relative reduction** — the ResNeXt-101 baseline leakage will likely differ from 27.8% (stronger backbone), so a bare absolute overlay is ambiguous.
- **Separability diagnostic (basis-free — interprets, does not gate).** Channel-wise rankings are meaningless under rotation; use a rotation-invariant measure: fit a linear firearm direction (receiver-logit probe) and a linear class subspace (class probe); report **principal angles** between them and **class-probe accuracy inside vs. in the orthogonal complement of the firearm subspace.** Disjoint (large principal angles; class recoverable only outside the firearm subspace) ⇒ dimensional separability (the capacity story); overlapping ⇒ still entangled. This explains *why* the frontier does or doesn't bend.
- **Durability:** refresh-test BEST incl. worst-case on the refreshed checkpoint (as §9.4), at **matched pre-refresh operating point** vs the V1 comparator.

**E2 decision.** If ResNeXt-101 bends the frontier, capacity is the lever → proceed to E3 (and the separability probe should show disjoint subspaces). If the wall persists at 4× width → firearm and class are **dimensionally inseparable in this feature**; scope the privacy claim to that, and **E3 is unlikely to help** (noise can't bury what isn't separable) — treat E3 as a short confirmation, not a campaign.

---

## 4. E3 — Combination (conditional on E2)

**Goal (narrow):** not "less leakage" — **durability**. Given capacity-separated class dimensions, bury them below the noise floor so a refresh can't resurrect them (V1's recall-preserving privacy was refresh-fragile because HSIC only *hides*).

- **VIB** (`privacy/vib.py`): Gaussian KL-rate term, `L = L_task + λ_H·HSIC + β·KL_rate`, `--vib-beta`. Prior `s²` fixed (see channel).
- **Channel — frozen-noise-floor (NOT held-SNR).** The audit showed that under a *held-SNR / re-measured-power* policy VIB is **scale-invariant and inert** (shrink E → μ², σ², s² all rescale, KL unchanged, nothing buried). For burial to be possible, VIB must shed power against a **fixed** noise level: compute `noise_std` **once** from the E2 Stage-A `coded_pwr` at the target SNR and **hold it fixed** through E3 training / Stage-C / utility eval. This is the V1-spec M7 "freeze" option, and it is *required* for E3 — not the optional knob the earlier draft made it. (`--noise-floor {frozen,held-snr}`, E3 uses `frozen`.)
- **σ>0 utility eval (M2):** privacy-side 48k evaluator with feature-level noise; **not** `main.py --evaluate` (pixel-noise). `eval_privacy` already taps post-channel features with feature-level noise (M1). **Fix the latent `device.index=None` σ>0 crash in `eval_privacy` before E3.**
- **Experiment:** at E2's recall-preserving BEST λ_H, sweep β at the target SNR; measure whether refresh-durability improves vs HSIC-only. **Gated on the E0-verify combo result** (`HANDOFF-v2-E0-verify.md` V3) — if the noise+HSIC complementarity doesn't survive a 2nd seed, E3 is a single confirmation run, not a sweep.
- **Explicit null ("VIB is just a second knob"):** VIB must add durability *at matched utility cost* vs. λ_H alone — not merely slide the frontier.

---

## 5. Metrics and measurement (apply the audit fixes)

- **Utility:** firearm **recall at matched FPR** (pin FPR ≤ ~2%, V1 BEST's operating point) + ROC-AUC + FPR/FP count. Un-pinned recall is invalid (V1 §9.4: recall is purchasable with false positives).
- **Privacy:** **pretrained-adv worst-case top-1 leakage over the full 48,800 background val** — primary. Kaiming optional.
- **Multi-seed is required for any "bend"/durability claim** (not "if time"): the effects are a few pp, and V1's same-checkpoint Stage-C band is ~1 pp (cross-campaign offset ~1.1 pp). Report mean±sd over ≥3 seeds at the BEST points and their comparators; at σ>0 also average over noise draws.
- **Durability — two numbers:** (1) absolute post-refresh worst-case leakage; (2) **gain-retention** = rebound measured against the *refreshed* λ_H=0/β=0 baseline **at the same SNR** (refresh that baseline too) — at σ>0 the channel caps everything, so comparing raw rebound to V1's σ=0 rebound flatters noise. Compare at matched pre-refresh operating points.
- **Threat model (pin in every σ>0 result):** single transmission; eavesdropper SNR = receiver SNR; adversary trains on fresh noisy draws. Record the two scope-limits (repetition-averaging defeats noise-privacy as 1/√n; a closer eavesdropper's SNR advantage does too — HSIC-privacy is immune to both). Add one robustness row at E3's BEST: Stage-C with eavesdropper SNR +6 dB and 4-draw averaging.
- **Stage-C convergence:** 30 epochs may under-converge at σ>0 (noisy features) — run 60 at the E3 BEST and confirm the number is stable.
- **Evaluator validation:** confirm the σ>0 utility path reproduces σ=0 numbers at `noise_std=None` before trusting any E3 utility number.

---

## 6. Code changes

- **`privacy/vib.py`** (new) — `gaussian_kl_rate(pre_channel, noise_std, prior_std)` closed form; σ=None guard. Unit-tested. (E3 only.)
- **`privacy/trainer.py`** — `joint_step` adds `β·KL_rate` (already threads `noise_std`); rate term on all samples, HSIC background-only.
- **`resnet_design2/my_resnet.py`** — confirm `resnext101_32x8d` Stage-A load path (E2). Frozen-noise-floor path for the channel (E3): `noise_std` fixed from Stage-A `coded_pwr`, not re-measured. Optional `--power-normalize` stays off/deferred.
- **`privacy/train_invariance.py`** — `--arch` already parametrized (E2 = `resnext101_32x8d`, σ=0, no code change beyond orchestration). For E3: `--vib-beta`, `--prior-std`, `--noise-floor frozen`, target-SNR path.
- **`privacy/eval_privacy.py`** — **fix the `device.index=None` σ>0 crash**; add the 48k feature-noise utility evaluator (M2); basis-free separability probe.
- **Scripts** — E2 sweep (mirror `run_hsic_refine.sh` on the new arch); E3 β-sweep; extend `parse_results` for new columns.

E2 is mostly orchestration on the existing HSIC pipeline; the new code (VIB, frozen-floor channel, separability probe, device-bug fix) is small and E3-scoped.

---

## 7. Out of scope (V2)

Standalone-noise ResNet-18 σ>0 (E1 — killed by Campaign D); GTGT-FM / K>0; learned VIB priors or per-element variance; CLUB; alternate tap layers (an open question for a later campaign — is the entanglement an artifact of the layer-2 ITIT tap?); formal DP. The held-SNR/power-normalized VIB variant is not used (inert for burial).

---

## 8. Known risks / honest caveats

- **E2 may just show the same wall at 4× width** — a real, publishable negative ("firearm-vs-class is dimensionally inseparable in this feature"), and the primary thing V2 exists to settle. Don't over-invest in E3 until E2's frontier and separability probe say capacity separated the tasks.
- **Single seed is not enough** for few-pp claims — multi-seed at BEST points is required (§5).
- **E3's premise is thin** — it rests on E2 separability *and* the E0-verify combo confirmation; if either fails, E3 collapses to a short confirmation.
- **ResNeXt-101 is compute/I-O-heavy** (86.7M params, 512-ch code) — multi-hour runs; use the pretrained backbone.
- **σ>0 privacy is threat-model-contingent** (repetition, eavesdropper geometry) in a way HSIC privacy is not — any E3 durability win must be scoped accordingly.
