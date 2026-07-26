# V2 Design Spec — Capacity-First (with Noise/VIB only in the combination cell)

**Date:** 2026-07-25 (**revised 2026-07-26** after Campaign D)
**Status:** Re-scoped capacity-first (supersedes the noise-first draft); implementation plan to follow.
**Branch (base):** `dev/invariance-privacy-hsic-v1`
**Predecessor spec:** `docs/superpowers/specs/2026-07-23-invariance-privacy-hsic-vib-design.md` (V1: HSIC, ITIT, σ=0)
**Motivating results:** `docs/superpowers/results/2026-07-25-invariance-hsic-resnet18-results.md` §9 (V1 frontier), `docs/superpowers/results/2026-07-26-v2e0-snr-window-resnet18.md` (Campaign D / E0), `docs/EXPERIMENT-LOG.md`

> **Revision note.** The original V2 draft led with a "noise-first" ResNet-18 σ>0 experiment (E1) on the hypothesis that channel noise is a privacy lever *independent* of representational capacity. **Campaign D (E0) tested that premise on frozen V1 checkpoints and returned a null.** The audit follow-up (`HANDOFF-v2-E0-verify.md`, results §6) then closed it firmly and surfaced one protocol finding that ripples backward:
> - **The worst-case adversary under-converges at 30-epoch Stage-C — for *every* config — by +3.5–5.4 pp** (it keeps learning past 30 ep). All prior 30-ep leakage numbers (including V1 §9's frontier) *understate* the worst case by that much; the effect is roughly symmetric so *relative* comparisons hold, but **60-epoch (or convergence-gated) Stage-C is now the standard**, and any cross-campaign overlay must be matched-convergence.
> - At **converged (60-ep), matched-FPR** utility, HSIC λ=30 (**22.14%**) and noise −7 dB (**21.77%**, recall 0.993) are a **dead tie** — no recall-preserving window survives; the noise lever is confirmed *not* independent.
> - **The E3 "noise+HSIC complementarity" was a 30-ep artifact** — seed-robust but under-converged; at 60 ep it lands on the frontier with everything else. **E3 must not lean on a complementarity premise.**
>
> So **the standalone-noise E1 arm is dropped**, and E3 is demoted to a short negative-check (§4). The capacity-first decision holds regardless.

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
| **E2** | ResNeXt-101, σ=0, HSIC | Does 4× capacity **bend the frontier** — lower worst-case leakage at the *recall-preserving* operating point? | **Primary** |
| **E3** | ResNeXt-101, σ>0, HSIC+VIB | *Only if E2 bends:* can a frozen-noise-floor VIB make the (now-separable) recall-preserving privacy **refresh-durable**? | Conditional — short check |

Everything else is held identical to V1 §9 so results overlay: ITIT (`--GT-alg 1 --background-K 0`), PK sampler, and the Stage-C protocol (pretrained-adv worst-case primary), **now at 60-epoch/convergence-gated Stage-C**.

**E2 runs gated, cheapest-first (the E0 philosophy):** a single recall-preserving *operating point* answers the go/no-go; the full frontier + durability run only if that point clears the bar.

---

## 3. E2 — Capacity: gated, single operating point first (primary)

**Arch:** `resnext101_32x8d` — layer-2 code 512 ch (4× ResNet-18), ImageNet-pretrained weights (wired in `resnet_design2/my_resnet.py`) to skip a multi-day backbone pretrain. **Only** this lever changes vs V1 §9.

**The whole capacity frontier collapses to one number for the go/no-go:** worst-case leakage at the **recall-preserving edge** (recall 50/50, matched FPR≤2%, **60-ep converged**). ResNet-18's value there is **~22%** (HSIC λ=30, 60-ep — already the converged comparator, so the comparison is matched-convergence). So E2's gate is:

> On ResNeXt-101, at recall = 50/50 (matched FPR, 60-ep Stage-C), is worst-case leakage **meaningfully below ~22%** (beyond the ~1 pp cross-campaign band)?

### E2.1 — The gate (single operating point, ~3 runs)
- **Stage-A pretrain** (once) — fine-tune from the ImageNet-pretrained ResNeXt-101 backbone; ~hours, not multi-day.
- **Bracket the recall-preserving edge, don't sweep.** A blind single λ_H can miss (too weak → leakage unchanged; too strong → recall drops below 50/50), so bracket: ResNeXt-101 has more capacity, so its recall-preserving λ_H is likely *higher* than ResNet-18's 30 — start λ_H ≈ 50–100, measure recall@2%FPR, adjust once or twice to sit at the 50/50 edge (**~3 invariance runs**, not the ~10-point frontier).
- **Read the gate:** at the largest λ_H holding recall 50/50, worst-case leakage (pretrained-adv, **60-ep** Stage-C, mean±sd over ≥3 seeds).
- **Corroborate free — basis-free separability probe** on that checkpoint *and* on a V1 ResNet-18 checkpoint (the entangled reference): fit a linear firearm direction (receiver-logit probe) and a linear class subspace (class probe); report **principal angles** and **class-probe accuracy inside vs. in the orthogonal complement of the firearm subspace.** Disjoint (large angles; class recoverable only outside the firearm subspace) ⇒ dimensional separation. This interprets; the leakage-at-recall number adjudicates.

**Gate decision.** Leakage meaningfully < ~22% at recall 50/50 → capacity bent the wall → go to E2.2. Leakage ≈ 22% → wall persists at 4× width → **stop the capacity campaign**; scope the privacy claim to "firearm/class are dimensionally inseparable in this feature," and pivot to the deeper open question (§7: is the entanglement a layer-2-tap / K=0 artifact?). E3 does not run (noise can't bury what isn't separable).

### E2.2 — Characterize (only if the gate clears)
Then, and only then, run the fuller λ_H frontier (coarse {0,1,10,100,1000} → refine the knee), **pretrained-adv 60-ep Stage-C for every point**, and the refresh-durability test at BEST (worst-case on the refreshed checkpoint, matched pre-refresh operating point). **Report per-arch baseline + both absolute and relative reduction** (the ResNeXt-101 baseline leakage will differ from ResNet-18's — a bare absolute overlay is ambiguous), all at matched 60-ep convergence.

---

## 4. E3 — Combination: a short negative-check (conditional on E2 bending)

**Status: demoted.** E3 originally rested on the noise+HSIC "complementarity" hint, which **the E0 verification killed** (it was a 30-ep convergence artifact; at 60 ep it lands on the frontier — see the revision note). So E3 is no longer a β-sweep campaign. It runs **only if E2.1's gate clears** (capacity actually separates the tasks), and then only as a **single-operating-point durability check**: does a frozen-noise-floor VIB make the *separated* recall-preserving privacy survive a refresh where HSIC-alone's did not?

- **VIB** (`privacy/vib.py`): Gaussian KL-rate term, `L = L_task + λ_H·HSIC + β·KL_rate`, `--vib-beta`. Prior `s²` fixed.
- **Channel — frozen-noise-floor (NOT held-SNR), required.** Under a held-SNR / re-measured-power policy VIB is **scale-invariant and inert** (shrink E → μ², σ², s² all rescale, KL unchanged, nothing buried). Burial needs a **fixed** noise level: compute `noise_std` **once** from the E2 Stage-A `coded_pwr` at the target SNR and hold it fixed through training / Stage-C / utility eval (V1-spec M7 "freeze"). `--noise-floor frozen`.
- **σ>0 utility eval (M2):** privacy-side 48k feature-noise evaluator; **not** `main.py --evaluate` (pixel-noise). **Fix the latent `device.index=None` σ>0 crash in `eval_privacy` first.**
- **The check (one point, not a sweep):** at E2's recall-preserving BEST λ_H + one target SNR, one β chosen to sit at the same recall, refresh-test, compare **gain-retention** (§5) vs HSIC-alone at the matched pre-refresh operating point. Explicit null: VIB must add durability *at matched utility cost*, not merely slide the frontier. If it doesn't, note it and stop — do not expand into a β-sweep on a dead complementarity premise.

---

## 5. Metrics and measurement (apply the audit fixes)

- **Utility:** firearm **recall at matched FPR** (pin FPR ≤ ~2%, V1 BEST's operating point) + ROC-AUC + FPR/FP count. Un-pinned recall is invalid (V1 §9.4: recall is purchasable with false positives).
- **Privacy:** **pretrained-adv worst-case top-1 leakage over the full 48,800 background val** — primary. Kaiming optional.
- **Multi-seed is required for any "bend"/durability claim** (not "if time"): the effects are a few pp, and V1's same-checkpoint Stage-C band is ~1 pp (cross-campaign offset ~1.1 pp). Report mean±sd over ≥3 seeds at the BEST points and their comparators; at σ>0 also average over noise draws.
- **Durability — two numbers:** (1) absolute post-refresh worst-case leakage; (2) **gain-retention** = rebound measured against the *refreshed* λ_H=0/β=0 baseline **at the same SNR** (refresh that baseline too) — at σ>0 the channel caps everything, so comparing raw rebound to V1's σ=0 rebound flatters noise. Compare at matched pre-refresh operating points.
- **Threat model (pin in every σ>0 result):** single transmission; eavesdropper SNR = receiver SNR; adversary trains on fresh noisy draws. Record the two scope-limits (repetition-averaging defeats noise-privacy as 1/√n; a closer eavesdropper's SNR advantage does too — HSIC-privacy is immune to both). Add one robustness row at E3's BEST: Stage-C with eavesdropper SNR +6 dB and 4-draw averaging.
- **Stage-C convergence — 60 epochs (or convergence-gated) is the standard, not 30.** The E0 verification found the worst-case adversary keeps learning past 30 ep on *every* config (+3.5–5.4 pp at 60 ep). All E2/E3 leakage numbers use 60-ep Stage-C; **the ResNet-18 comparator (~22% at recall 50/50) is already the 60-ep number**, so the overlay is matched-convergence. Any comparison to a V1 §9 *30-ep* number must first be re-run at 60 ep (they understate by ~4–5 pp).
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
