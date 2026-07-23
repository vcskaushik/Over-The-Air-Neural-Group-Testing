# Invariance-Based Privacy Training (HSIC + VIB) — Design Spec

**Date:** 2026-07-23
**Status:** Design approved; implementation plan to follow.
**Scope:** A new, non-adversarial training path for the OTA-NGT privacy problem that
replaces the failed min-max adversarial encoder fine-tune with an *invariance* objective:
train the encoder (and receiver) so post-channel features are statistically independent of
the fine-grained ImageNet class while preserving binary firearm-detection utility. v1 covers
**ITIT only**.

**Related docs:**
- Original design: `docs/superpowers/specs/2026-04-17-privacy-preserving-ota-ngt-design.md`
- First-pass results (adversarial v1 failure): `docs/superpowers/results/2026-04-18-itit-resnet18-no-noise-results.md`
- Conceptual write-up + review-hardened framing: temporary section in `Privacy_Preserving_OTA_NGT.tex`

---

## 1. Motivation

The adversarial min-max fine-tune (existing `privacy/trainer.py:stage_b_step`) failed: its
privacy "gain" was obfuscation, not erasure — it pushed *up* a lower bound on leakage
`I(z̃; c)`, so a fresh adversary (or a brief utility refresh) recovered the class information
(35 → 29 → 35% top-1). The fix is to optimize an objective whose reduction genuinely reduces
information:

- **HSIC** — a non-parametric statistical-independence penalty targeting `I(z̃; c)` directly,
  at fixed transmitted power. Primary mechanism.
- **VIB rate** — the Gaussian-channel KL/rate term. Blind (rotation-invariant power lever) but
  *physically irreversible* below the channel noise floor; only meaningful with `σ > 0`.
  Secondary/optional mechanism.

Both are **scramble-invariant** (independence and rate are preserved under bijective
re-encoding), which is exactly the property the learned adversary lacked.

---

## 2. Key design decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Mechanisms in v1 | **HSIC (primary) + VIB rate (optional)**. CLUB deferred. | HSIC is the "decisive experiment"; CLUB is expected to reproduce v1 fragility. |
| Training arm | **Joint E+R**, warm-started from a Stage-A checkpoint. No frozen receiver. | Freezing R was an anti-scrambling device for a *learned* adversary. HSIC/VIB are scramble-invariant, so a co-adapting R cannot manufacture a scramble that lowers them. Freezing is not load-bearing. |
| HSIC feature summary | **Frozen ImageNet-`pretrained=True` `layer3+layer4+avgpool`** extractor, optionally ensembled with 1–2 **fixed random-conv** projections. **Never trainable.** | The eval adversary is convolutional; the summary must be spatially aware. A pretrained class-feature extractor is threat-faithful; frozen ⇒ non-gameable, no moving target. Trainable ⇒ reintroduces min-max (max-HSIC critic) or degenerate collapse (joint-min). |
| Batch sampling | **PK-balanced background sampler** + firearm samples per batch. | HSIC's delta label kernel needs same-class pairs; at batch 32 over ~976 classes, expected collisions ≈ 0.5/batch, degenerating HSIC into a label-blind (compression) statistic. |
| Entry point | **New `privacy/train_invariance.py`**. | Leaves the v1 adversarial path untouched (mirrors "no modifications to main.py" principle). |
| GT algorithm | **ITIT only** (`--GT-alg 1`, `--background-K 0`). | Cleaner delta-kernel/sampler story; GTGT-FM label kernel + channel-sum semantics deferred. |

---

## 3. Architecture and data flow

Reuses the existing `ResNet_GT` split. Components:

- **`E` (encoder):** `conv1 → bn1 → relu → maxpool → layer1 → layer2`. Trained.
- **`R` (receiver):** `layer3 → layer4 → avgpool → fc(2)`. Trained (jointly with E — **not** frozen).
- **HSIC extractor `Φ` (new, frozen):** ImageNet-pretrained `layer3 → layer4 → avgpool` producing a
  512-d vector, optionally plus 1–2 fixed random-conv projection heads. `requires_grad=False`, `eval()`.
- **Adversary `A′`:** exists **only** in Stage C (`eval_privacy.py`), trained from scratch to measure leakage.

Per-minibatch flow (joint step):
```
images ──E──▶ pre = E(x)
pre ──channel(noise_std=σ)──▶ z̃ = (Σ_K pre + n)/K        # ITIT: K=1 ⇒ z̃ = E(x) + n
z̃ ──R──▶ util_logits              ── L_task = CE(util_logits, y_firearm)      [all samples]
z̃[bg] ──Φ (frozen)──▶ feat        ── L_hsic = Σ_extractors HSIC(feat, c)      [background only]
pre, σ ──▶ L_rate = KL(N(pre,σ²I) ‖ N(0,s²I))                                 [all samples, σ>0 only]
total = L_task + λ_H · L_hsic + β · L_rate
```
One optimizer step updates `E + R`. `Φ` and (in v1) the channel are not updated.

---

## 4. New modules

### 4.1 `privacy/hsic.py`
`class HSICPenalty(nn.Module)`:
- **Extractors:** built from `arch_name`. `pretrained` extractor reuses the `AdversaryHead` decoder
  stack with `pretrained=True`, frozen. Random extractors: fixed random-init small conv stacks
  (frozen) mapping `(512,28,28) → pooled vector`. Config selects `pretrained | random | both` and
  the number of random heads.
- **Estimator:** biased V-statistic `HSIC = tr(K H L H) / (m-1)²` with centering `H = I − 11ᵀ/m`.
- **Feature kernel `K`:** RBF on each extractor's pooled vector; **multi-bandwidth** — median-heuristic
  `h₀` per batch per extractor, summed over a configurable multiplier set (default `{0.5, 1, 2}·h₀`).
- **Label kernel `L`:** delta kernel `1[c_i = c_j]` on fine-grained class.
- **Background masking:** firearm rows removed before kernels (firearm leakage is intended).
- **Output:** scalar penalty = sum over extractors and bandwidths. Differentiable w.r.t. `z̃` (hence `E`);
  extractors receive no gradient.

### 4.2 `privacy/vib.py`
`gaussian_kl_rate(pre_channel, noise_std, prior_std) -> Tensor`:
- Closed form of `KL(N(μ, σ²I) ‖ N(0, s²I))` with `μ = pre_channel` (the encoder output),
  `σ = noise_std`, `s = prior_std`, summed over feature dimensions, meaned over the batch.
- Guard: raise `ValueError` if called with `noise_std is None` (σ=0 ⇒ degenerate KL). The training
  script asserts `--SNR` is set whenever `--vib-beta > 0`.

### 4.3 `privacy/sampler.py`
`class PKBackgroundSampler(Sampler)`:
- Yields batches of `P` distinct background classes × `K` samples each, **plus** `F` firearm samples.
- Requires an index of sample → (class, is_firearm) from `GroupTestingDataset`.
- Defaults `P=8, K=4, F=4` (batch 36). Exposes `pk_classes`, `pk_per_class`, `pk_firearm`.

---

## 5. Changes to existing code

- **`privacy/trainer.py`** — add `joint_step(...)`: forward `E → channel → R` for `L_task`; compute
  HSIC on background post-channel features via the frozen extractor; compute VIB KL from `pre` and σ;
  `total = L_task + λ_H·L_hsic + β·L_rate`; one `E+R` optimizer step; return per-term metrics.
  The existing `stage_b_step` is left unchanged.
- **`privacy/train_invariance.py`** (new) — entry point mirroring `train_privacy.py`: arg parsing,
  data loading with `PKBackgroundSampler`, warm-start `E+R` from `--stage-a-ckpt`, run joint training,
  write `invariance_final.pth.tar` in main.py-compatible format (key rename + `module.` prefix, as the
  existing conversion does). Shared helpers (checkpoint load, data root wiring) factored into
  `privacy/_common.py` if duplication is non-trivial; otherwise inlined.
- **`privacy/eval_privacy.py`** — reused for Stage C. Add optional `--adv-init {kaiming, pretrained}`
  (default `kaiming`) so a worst-case pretrained-initialized adversary can be measured too.
- **`resnet_design2/my_resnet.py`** — only if needed: ensure `joint_step` can obtain the pre-channel
  tensor (`encode`) and post-channel tensor (`channel`) separately. The existing `encode`/`channel`/
  `decode` methods already expose these; no change expected.

---

## 6. Configuration flags (defaults)

| Flag | Default | Notes |
|---|---|---|
| `--stage-a-ckpt` | required | Warm-start `E+R`. |
| `--hsic-lambda` | `1.0` | `0` disables HSIC. |
| `--vib-beta` | `0.0` | `0` disables VIB rate. `>0` requires `--SNR`. |
| `--hsic-extractor` | `pretrained` | `pretrained | random | both`. |
| `--hsic-num-random` | `2` | Random-conv heads when `random`/`both`. |
| `--hsic-bandwidths` | `0.5,1,2` | Median-heuristic multipliers. |
| `--prior-std` | `1.0` | VIB prior `s`. |
| `--pk-classes` / `--pk-per-class` / `--pk-firearm` | `8 / 4 / 4` | Sampler `P / K / F`. |
| `--enc-lr` / `--rec-lr` | `1e-3` / `1e-3` | E and R learning rates (joint). |
| `--epochs` | `30` | Joint training epochs. |
| `--SNR`, `--phase` | reused | Channel params; must match the Stage-A checkpoint. |
| `--GT-alg`, `--background-K` | `1`, `0` | ITIT only in v1 (asserted). |

Existing shared flags (`--data`, `--arch`, `--output_dir`, workers, seed, print-freq) reused.

---

## 7. Metrics and evaluation

- **Utility** (reuse `main.py validate()` on the 48k protocol): Firearm Acc@1, ROC-AUC, firearm recall.
- **Privacy** (Stage C, `eval_privacy.py`): fresh adversary top-1 ImageNet accuracy over the 979-way
  reorganized label space (chance ≈ 0.1%). Optional pretrained-init arm for worst case.
- **Refresh diagnostic (R4):** after joint training, optionally run a brief utility-only fine-tune
  (`main.py`) then re-run Stage C. Reported as a **deployment-lifecycle fragility** number, not as the
  honest leakage of the deployed encoder. The honest number is Stage C on the deployed (unrefreshed)
  encoder.

---

## 8. Testing (TDD)

New tests mirror the style of existing `tests/`:
- **HSIC** (`test_hsic.py`): ≈0 for independent `z⊥c`, strictly >0 for dependent; **spatial-sensitivity**
  — class info encoded only in spatial layout (constant channel means) is detected by the extractor-based
  HSIC but missed by a global-pool baseline (proves the R1 mitigation); multi-bandwidth sums correctly;
  background masking removes firearm rows; extractor params receive no gradient.
- **VIB** (`test_vib.py`): matches hand-computed closed form on a small tensor; ≥0; zero at `μ=0, σ=s`;
  `noise_std=None` raises.
- **Sampler** (`test_sampler.py`): every emitted batch has exactly `P` classes × `K` background samples
  plus `F` firearm samples; background classes are distinct within a batch.
- **Joint step** (extend `test_trainer.py`): HSIC and VIB gradients reach `E`; task gradient reaches
  `E` and `R`; frozen extractor accumulates no gradient; loss terms are finite.
- **Integration smoke** (`test_smoke_invariance.py`): a few joint iterations on tiny synthetic data run
  end-to-end and produce a loadable checkpoint.

---

## 9. Known conceptual risks carried into implementation (R1–R5)

These are documented so the implementation and reporting stay honest; they are mitigations, not proofs.

- **R1 — Spatial blindness.** Mitigated by the frozen conv extractor (§4.1). Do **not** claim "no
  scrambling escape"; Stage C is the arbiter.
- **R2 — Delta-kernel / small-batch degeneracy.** Mitigated by the PK sampler (§4.3).
- **R3 — Static-critic overfitting.** A fixed extractor can be partly evaded; mitigate with the
  extractor ensemble and multi-bandwidth kernels. **Never report `HSIC` as the privacy number** — only
  Stage C.
- **R4 — Refresh vs. passive threat model.** Refresh is a lifecycle diagnostic, not the honest number
  (§7).
- **R5 — Conditional floor.** The fundamental limit is `I(c; T | y_bin=0)` (background-restricted);
  near-floor Pareto resolution is limited by the 50 firearm val images and needs multi-seed runs.

---

## 10. Out of scope (v1)

- CLUB and any trainable/adversarial HSIC projection.
- GTGT-FM (group transmission) HSIC label kernel and channel-sum invariance semantics.
- The "slow-critic re-snapshot" extractor variant.
- Learned per-channel VIB variance / learned prior (fixed isotropic prior only in v1).
- Formal DP guarantees; backbones beyond the existing `--arch` support (code is arch-agnostic, but the
  paper claim scope is unchanged).
