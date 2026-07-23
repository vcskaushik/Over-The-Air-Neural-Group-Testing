# Invariance-Based Privacy Training (HSIC) — Design Spec (v1)

**Date:** 2026-07-23
**Status:** Design approved; hardened after spec review; implementation plan to follow.
**Scope:** A new, non-adversarial training path for the OTA-NGT privacy problem that replaces
the failed min-max adversarial encoder fine-tune with an *invariance* objective. **v1 covers
HSIC only, ITIT only, at σ=0 (no channel noise).** The VIB rate term and everything the σ>0
channel requires are explicitly deferred to a **v2 spec** (see §10).

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
information. **HSIC** is a non-parametric statistical-independence penalty targeting `I(z̃; c)`
directly, at fixed transmitted power, and it is **scramble-invariant** (independence is
preserved under bijective re-encoding) — the property the learned adversary lacked.

VIB (the Gaussian-channel rate term) is the natural companion for σ>0 but is deferred: it is
inert at σ=0, and making it honest requires channel-model surgery that is out of scope here (§10).

---

## 2. Key design decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Mechanism in v1 | **HSIC only**. VIB deferred to v2. CLUB deferred. | HSIC is the "decisive experiment"; VIB is inert at σ=0; CLUB is expected to reproduce v1 fragility. |
| Channel noise | **σ=0 (no noise)** in v1. | Matches the ResNet-18 first pass for comparison; avoids the σ>0 channel-consistency work (§10). At σ=0, `channel()` returns `z̃ = E(x)` (post == pre), so existing pre-channel eval conventions are consistent. |
| Training arm | **Joint E+R**, warm-started from a Stage-A checkpoint. No frozen receiver. | Freezing R was an anti-scrambling device for a *learned* adversary. HSIC is scramble-invariant, so a co-adapting R cannot manufacture a scramble that lowers it. Freezing is not load-bearing. |
| HSIC feature summary | **Frozen ImageNet-`pretrained=True` `layer3+layer4+avgpool`** extractor, optionally ensembled with 1–2 **fixed random-conv** projections. **Never trainable.** In/out channels **derived from `--arch`** (ResNet-18 layer2 = 128ch, ResNeXt-101 = 512ch). | The eval adversary is convolutional; the summary must be spatially aware. A pretrained class-feature extractor is threat-faithful; frozen ⇒ non-gameable, no moving target. Trainable ⇒ min-max or degenerate collapse. |
| Batch sampling | **PK-balanced background sampler** + firearm samples per batch, over the **full** background pool. | HSIC's delta label kernel needs same-class pairs; at batch 32 over ~976 classes, expected collisions ≈ 0.5/batch, degenerating HSIC into a label-blind statistic. |
| Entry point | **New `privacy/train_invariance.py`**. | Leaves the v1 adversarial path untouched. |
| GT algorithm | **ITIT only** (`--GT-alg 1`, `--background-K 0`). | Cleaner delta-kernel/sampler story; GTGT-FM deferred. |

---

## 3. Architecture and data flow

Reuses the existing `ResNet_GT` split. Components:

- **`E` (encoder):** `conv1 → bn1 → relu → maxpool → layer1 → layer2`. Trained.
- **`R` (receiver):** `layer3 → layer4 → avgpool → fc(2)`. Trained jointly with E (**not** frozen).
- **HSIC extractor `Φ` (new, frozen):** ImageNet-pretrained `layer3 → layer4 → avgpool` producing a
  pooled vector, optionally plus 1–2 fixed random-conv projection heads. `requires_grad=False`, `eval()`.
- **Adversary `A′`:** exists **only** in Stage C (`eval_privacy.py`), trained from scratch to measure leakage.

Per-minibatch flow (joint step, σ=0):
```
images ──E──▶ pre = E(x)
pre ──channel(noise_std=None)──▶ z̃ = pre            # ITIT σ=0: z̃ = E(x) exactly
z̃ ──R──▶ util_logits            ── L_task = CE(util_logits, y_firearm)     [all samples]
z̃[bg] ──Φ (frozen)──▶ feat      ── L_hsic = mean_extractors HSIC(feat, c)  [background only]
total = L_task + λ_H · L_hsic
```
One optimizer step updates `E + R`. `Φ` is never updated.

**Convention note (important for implementers):** the existing ITIT code (`trainer.py:86`,
`eval_privacy.py:110/146`) feeds the adversary the **pre-channel** tensor `pre`, discarding `post`.
At σ=0 this is identical to post-channel. The invariance path standardizes on **post-channel `z̃`**
for both HSIC (training) and the Stage-C adversary, so that the v2 σ>0 work is a drop-in. Implement
HSIC and Stage-C input as `post`, not `pre`.

---

## 4. New modules

### 4.1 `privacy/hsic.py`
`class HSICPenalty(nn.Module)`:
- **Extractors** built from `arch_name` with **arch-derived in-channels** (do not hardcode 512):
  the `pretrained` extractor reuses the `AdversaryHead` decoder stack with `pretrained=True`, frozen,
  `eval()`. Random extractors are fixed random-init conv stacks (frozen) mapping the layer2 tensor to a
  pooled vector. Config selects `pretrained | random | both` and the random-head count.
- **Estimator:** biased V-statistic `HSIC = tr(K H L H) / (m-1)²`, centering `H = I − 11ᵀ/m`.
- **Masking order (invariant):** firearm rows are removed **first**; the bandwidth, centering, and
  estimator all operate on the background-only submatrix. With the PK sampler, post-mask `m = P·K` is
  **constant** across batches (a design invariant the estimator relies on).
- **Feature kernel `K`:** RBF on each extractor's pooled vector; **multi-bandwidth** — per-batch,
  per-extractor median heuristic `h₀ = max(median pairwise distance, ε)` (ε-guard against local feature
  collapse), summed over configurable multipliers (default `{0.5, 1, 2}·h₀`).
- **Label kernel `L`:** delta kernel `1[c_i = c_j]` on fine-grained class.
- **Normalization:** the penalty is the **mean** over extractors and bandwidths (not sum), so `λ_H` is
  comparable across `--hsic-extractor` and bandwidth-count choices.
- **Output:** scalar, differentiable w.r.t. `z̃` (hence `E`); extractors receive no gradient.

### 4.2 `privacy/sampler.py`
`class PKBackgroundSampler(Sampler)`:
- Yields **index lists** (passed to the loader as `batch_sampler=`, which supersedes
  `batch_size/shuffle/drop_last`): each batch = `P` distinct background classes × `K` samples + `F`
  firearm samples. Defaults `P=8, K=4, F=4` (batch 36).
- Built from a sample→(wnid, is_firearm) index derived from `PrivacyTaskCoalitionDataset.dataset_samples[0]`
  (wnid = parent dir of the path; firearm via the target field). No `GroupTestingDataset` attribute is assumed.
- **Underfilled-class policy:** classes with `< K` background samples are excluded from PK selection
  (logged). **Epoch length** = `⌈N_bg / (P·K)⌉` batches, where `N_bg` is the usable background pool.
- Requires the dataset change in §5 (full background pool); otherwise ~43% of classes cannot fill a block.

---

## 5. Changes to existing code

- **`privacy/dataset.py`** — add an invariance mode that exposes the **full** background sample list
  instead of truncating negatives to `len(positives)` (`dataset.py:52`). The firearm/background ratio in
  training is then controlled by the sampler's `F`, not by dataset construction. Existing behavior for the
  v1 adversarial path is preserved (gated by a flag/arg).
- **`privacy/adversary.py`** — add a `pretrained: bool = False` constructor arg (currently hardcoded
  `False` at `adversary.py:23`). Needed by both the HSIC extractor (`pretrained=True`) and the Stage-C
  worst-case arm.
- **`privacy/trainer.py`** — add `joint_step(...)`: forward `E → channel(noise_std=None) → R` for
  `L_task`; compute HSIC on **post-channel** background features via the frozen extractor; one `E+R`
  optimizer step on `L_task + λ_H·L_hsic`; return per-term metrics + per-extractor HSIC. Existing
  `stage_b_step` unchanged.
- **`privacy/train_invariance.py`** (new) — entry point mirroring `train_privacy.py`: arg parsing, data
  loading with `PKBackgroundSampler`, warm-start `E+R` from `--stage-a-ckpt`, joint training, write
  `invariance_final.pth.tar`. **Checkpoint format** (enumerated, matching `main.py --resume` at
  `main.py:323,332`): `state_dict` with `module.` prefix, `coded_pwr` (carried through from the Stage-A
  checkpoint unchanged in v1), `epoch`. **Startup diagnostic (mandatory):** on the warm-start weights,
  compute `HSIC(Φ(z̃), c)` on a few real batches and compare against a permuted-label baseline; assert/log
  it is clearly above baseline, so a distribution-mismatched (inert) extractor is caught before training.
- **`privacy/eval_privacy.py`** — Stage C. (a) evaluate leakage on **all background val images**, not the
  ~300-sample `PrivacyTaskCoalitionDataset` val (`__len__` = 2×positives); (b) add optional
  `--adv-init {kaiming, pretrained}` (default `kaiming`) for a worst-case pretrained adversary; (c) at σ=0
  the pre/post distinction is moot, but standardize the adversary input on `post` per §3.
- **`resnet_design2/my_resnet.py`** — no change expected; `encode`/`channel`/`decode` already expose the
  separable tensors §3 needs (verified).

---

## 6. Configuration flags (defaults)

| Flag | Default | Notes |
|---|---|---|
| `--stage-a-ckpt` | required | Warm-start `E+R`. |
| `--hsic-lambda` | `1.0` | `0` disables HSIC (**mandatory control run**, §7). Default is a placeholder — sweep; biased HSIC at m=32 is small, so the useful range is likely ≫1. |
| `--hsic-extractor` | `pretrained` | `pretrained | random | both`. |
| `--hsic-num-random` | `2` | Random-conv heads when `random`/`both`. |
| `--hsic-bandwidths` | `0.5,1,2` | Median-heuristic multipliers. |
| `--pk-classes` / `--pk-per-class` / `--pk-firearm` | `8 / 4 / 4` | Sampler `P / K / F`. |
| `--enc-lr` / `--rec-lr` | `1e-4` / `1e-4` | Joint **fine-tune** LRs (not the 1e-3 from-scratch LR — protects the warm start). |
| `--epochs` | `30` | Joint training epochs. |
| `--GT-alg`, `--background-K` | `1`, `0` | ITIT only in v1 (asserted). |
| `--SNR` | unset / `None` | σ=0 in v1; a set `--SNR` is a config error in v1 (asserted; belongs to v2). |

Existing shared flags (`--data`, `--arch`, `--output_dir`, workers, seed, print-freq) reused.

---

## 7. Metrics and evaluation

- **Utility** (reuse `main.py validate()` on the 48k protocol — valid at σ=0 since main.py's channel
  degenerates to noiseless): Firearm Acc@1, ROC-AUC, firearm recall.
- **Privacy** (Stage C, `eval_privacy.py`): fresh adversary top-1 ImageNet accuracy over the 979-way
  label space, evaluated on **all background val images** (chance ≈ 0.1%). Optional pretrained-init arm.
- **Mandatory control (M4):** a `λ_H=0` run **with the PK sampler and joint E+R** is the reference for
  both utility and leakage. Without it, sampler-induced distribution shift (PK batches are ~11% firearm
  vs. the 50/50 training the first pass used) confounds every reported delta.
- **Refresh diagnostic (R4):** after joint training, optionally run a brief utility-only `main.py`
  fine-tune (itself a 50/50-distribution jump — interpret accordingly), then re-run Stage C. Reported as a
  **deployment-lifecycle fragility** number, not the honest leakage of the deployed encoder.

---

## 8. Testing (TDD)

- **HSIC** (`test_hsic.py`): ≈0 for independent `z⊥c`, strictly >0 for dependent; **spatial-sensitivity**
  — class info encoded only in spatial layout (constant channel means → global-pool baseline HSIC is
  exactly 0 after centering) is detected by the conv extractor; use the fixed random-conv extractor with a
  pinned seed and assert a *ratio* (extractor-HSIC ≫ pool-HSIC), not absolute thresholds; ε-bandwidth guard
  prevents div-by-zero on the collapsed-pool case; background masking removes firearm rows before bandwidth;
  extractor params receive no gradient.
- **Sampler** (`test_sampler.py`): every batch has exactly `P` distinct background classes × `K` + `F`
  firearm samples; underfilled classes excluded; epoch length correct.
- **Joint step** (extend `test_trainer.py`): HSIC gradient reaches `E`; task gradient reaches `E` and `R`;
  frozen extractor accumulates no gradient; grads *do* flow through Φ to E; loss terms finite.
- **Startup diagnostic** (`test_smoke_invariance.py` or a unit): the extractor-sanity check flags an inert
  (permuted-label-equivalent) extractor and passes on a class-informative one.
- **Integration smoke** (`test_smoke_invariance.py`): a few joint iterations on tiny synthetic data run
  end-to-end and produce a loadable, main.py-`--resume`-compatible checkpoint (has `state_dict`,
  `coded_pwr`, `epoch`).

---

## 9. Known conceptual risks carried into implementation (R1–R5)

Mitigations, not proofs. **Never report `HSIC` as the privacy number — only Stage C.**

- **R1 — Spatial blindness.** Mitigated by the frozen conv extractor (§4.1). Do not claim "no scrambling
  escape"; Stage C is the arbiter.
- **R2 — Delta-kernel / small-batch degeneracy.** Mitigated by the PK sampler over the full background pool.
- **R3 — Static-critic overfitting.** A fixed extractor can be partly evaded; mitigate with the ensemble +
  multi-bandwidth kernels; the epoch-0 diagnostic + Stage C catch inert/over-fit extractors.
- **R4 — Refresh vs. passive threat model.** Refresh is a lifecycle diagnostic, not the honest number (§7).
- **R5 — Conditional floor.** The fundamental limit is `I(c; T | y_bin=0)`; near-floor Pareto resolution is
  limited by the 50 firearm val images and needs multi-seed runs.

---

## 10. Out of scope (v1) → v2 spec

The **VIB rate term and the σ>0 channel** are deferred to a separate v2 spec. Making VIB honest requires
(all verified as real gaps in the current code):
- **M1** — Stage C must tap **post-channel** (noisy) features for ITIT (`eval_privacy.py` currently feeds
  `pre`); harmless at σ=0, wrong at σ>0.
- **M2** — the ITIT channel is inconsistent: `main.py` adds AWGN in **pixel space** while the privacy stack
  adds it at layer2. v2 must pick the feature-level channel as the system under test, warm-start from a
  matching Stage A, and provide a **privacy-side 48k utility evaluator** (cannot reuse `main.py validate()`
  at σ>0).
- **M7** — a **`coded_pwr` policy**: VIB shrinks feature power, so v2 must decide whether the noise floor is
  frozen at the Stage-A value (recommended — matches the "below the noise floor" irreversibility argument)
  or re-measured.

Also out of scope: CLUB; trainable/adversarial HSIC projection; GTGT-FM HSIC label kernel + channel-sum
semantics; the "slow-critic re-snapshot" extractor variant; learned per-channel VIB variance / learned
prior; formal DP guarantees.
