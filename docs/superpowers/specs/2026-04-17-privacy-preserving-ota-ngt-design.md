# Privacy-Preserving OTA-NGT — Design Spec

**Date:** 2026-04-17
**Status:** Design approved; implementation plan to follow.
**Scope:** New training mode for the OTA-NGT codebase that produces an encoder whose post-channel features are useful for the binary firearm-detection task but carry minimal information about fine-grained ImageNet class identity.

---

## 1. Motivation

The current OTA-NGT framework splits a ResNeXt across a noisy wireless channel: the user-side encoder transmits intermediate features and a base-station receiver classifies whether the (group of) image(s) contains a firearm. Anything an eavesdropper can recover from those over-the-air features is a privacy leak. We want a training procedure that **demonstrably reduces what an adversary can learn** about the input image's fine-grained ImageNet class while preserving binary firearm-detection utility.

This is a defender vs. eavesdropper problem with two key constraints that distinguish it from off-the-shelf adversarial-training recipes:

1. The adversary has the same architecture as the legitimate receiver (`layer3 → layer4 → avgpool → fc`) and has access to ground-truth ImageNet labels — i.e., a **worst-case** adversary.
2. Naive minimax training fails by **scrambling**: the encoder learns a bijective reparametrization that fools the *current* adversary while preserving information content. Co-training with the receiver hides the scrambling because the receiver learns the inverse code. Any honest measurement of leakage must defeat scrambling.

---

## 2. Threat model

| Aspect | Assumption |
|---|---|
| Adversary architecture | Identical to receiver: `layer3 + layer4 + avgpool + fc(K_adv)`, where `K_adv = 1000` for ITIT (CE) or `1000` sigmoid heads for GTGT-FM (BCE). |
| Adversary capabilities | Has ground-truth ImageNet labels. White-box access to the encoder during training (worst case). |
| Tap point | **Post-channel** — the same signal the receiver sees: encoder output + AWGN + optional random phase, plus (GTGT-FM only) the K-image sum. |
| Adversary objective | Recover the input image's ImageNet class (ITIT) or the K-hot indicator over the group's classes (GTGT-FM). |
| Out of scope | Active attacks (adversary modifies the channel), side-channel/metadata attacks, attacks on the firearm class itself (the system *intends* to leak that). |

The adversary's test-time leakage is measured as **top-1 ImageNet accuracy** (ITIT) or **mean per-class AUC** (GTGT-FM) of a freshly-trained adversary on the validation features.

---

## 3. Architecture

The defender's network is the existing `ResNet_GT`. We add one new module.

| Module | Role | Composition | Trained in |
|---|---|---|---|
| `E` (Encoder) | User-side feature extractor; output crosses the channel. | `conv1 → bn1 → relu → maxpool → layer1 → layer2`. Existing. | Stages A, B, A_recovery |
| `R` (Receiver) | Base-station firearm classifier. | `layer3 → layer4 → avgpool → fc(2)`. Existing. | Stages A, A_recovery (frozen in Stage B) |
| `A` (Adversary) | Eavesdropper — taps post-channel features, predicts ImageNet class. | `layer3 → layer4 → avgpool → fc(1000)` (CE) or `fc(1000)` + sigmoid (BCE for GTGT-FM). New. | Stages B (inner loop), C (from scratch) |

The post-channel feature tensor that all three downstream heads see is `(B, 512, 28, 28)` (for ResNeXt-101_32x8d at the layer2 split), regardless of which GT algorithm is in use. `A` and `R` consume this tensor identically; only their final FC heads differ.

---

## 4. Training pipeline

### Stage A — Utility pretrain (existing OTA-NGT)

End-to-end joint training of `E + R` on the binary firearm CE loss. This is the existing `main.py` training procedure with `--GT-alg 1` (ITIT) or `--GT-alg 2` (GTGT-FM); we reuse its checkpoint as the starting point for Stage B.

### Stage B — Privacy fine-tune (frozen receiver)

Receiver weights `R` are **frozen**. Adversary `A` and encoder `E` train jointly with two-time-scale updates per minibatch:

**Adversary inner loop** (`k_adv ≥ 1` SGD steps; default `k_adv = 5`). Each of the `k_adv` steps uses the **same** minibatch (post-channel features are cached after the first forward; only `A` is re-forwarded and back-propped):
```
A ← A − η_A · ∇_A L_adv
L_adv (ITIT)     = CE  ( A( post_channel( E(x) ) ),  y_imagenet )
L_adv (GTGT-FM)  = BCE ( A( post_channel( E(x) ) ),  K_hot(y_imagenet) )
```
The encoder is in `eval()` mode and `requires_grad=False` during these `k_adv` steps so gradients flow only into `A`. After the inner loop, the encoder is set back to `train()` and a fresh forward computes the encoder's outer gradient.

**Encoder outer step** (1 SGD step):
```
E ← E − η_E · ∇_E [ L_util  −  λ · L_priv ]
L_util = CE ( R( post_channel( E(x) ) ),  y_firearm )
```

`L_priv` is one of (selected by config flag `--priv-loss`):

- **`ce`** — `L_priv = − L_adv`. Negated adversary CE/BCE. Direct gradient reversal. Has a label-shift degenerate optimum (encoder can find a representation that systematically misroutes the adversary to one fixed wrong class).
- **`entropy`** — `L_priv = − H( softmax(A_logits) )` for ITIT, or `L_priv = − Σ_k H( σ(A_logits_k) )` for GTGT-FM. Pushes the adversary's posterior toward uniform / per-class 0.5. Bounded, no degenerate optimum.

Both formulations are first-class; the design-stage decision is to ablate them and report side-by-side curves.

**Why frozen R defangs scrambling.** With `R` fixed, the encoder's feature manifold is constrained to the preimage `R⁻¹(correct firearm class)`. Any reparametrization that doesn't preserve the receiver's expected input distribution explodes `L_util`. The encoder can only reduce leakage by *actually* removing information that wasn't needed for the firearm task — the privacy gain becomes measurable, not just a moving-target illusion.

### Stage A_recovery — Short utility refresh

Brief unfreezing of `R` after Stage B converges. Train `E + R` on `L_util` only for a small number of epochs (typically 1–3, configured by `--recovery-epochs`). Recovers utility regression introduced by Stage B without giving `R` enough time to fully re-learn the encoder's scrambled code (which would partially undo privacy).

### Stage C — Honest evaluation

Freeze both `E` and `R`. Initialize a **fresh** adversary `A′` from scratch and train it to convergence on the intercepted features over the training set. The honest leakage number is `A′`'s metric on the validation set. This step is what the paper reports; the Stage-B in-loop adversary is purely a privacy gradient signal.

If `A′`'s leakage exceeds the privacy budget, raise `λ` and re-run Stage B → A_recovery → C. (The full λ-sweep is the trade-off curve.)

---

## 5. Hyperparameters and defaults

| Knob | Default | Notes |
|---|---|---|
| `--priv-loss` | `entropy` | `ce` available as ablation. |
| `--lambda` | grid: `{0.0, 0.1, 0.3, 1.0, 3.0, 10.0}` | One run per λ; `λ = 0.0` is the no-privacy baseline. |
| `--k-adv` | `5` | Adversary inner-loop steps per encoder step. |
| `--adv-lr` | `1e-3` | Adversary SGD learning rate; constant. |
| `--enc-lr` | inherited from existing `--lr` (default `1e-3`) | Encoder SGD; reuses existing `main.py` schedule. |
| `--stage-b-epochs` | `30` | Stage-B epochs; tune empirically. |
| `--recovery-epochs` | `2` | Stage A_recovery epochs. |
| `--stage-c-epochs` | `60` | Fresh adversary trained from scratch in Stage C. |
| Channel parameters | reuse existing `--SNR`, `--phase`, `--snr-schedule`, `--snr-type` | Stage B and Stage C must use the **same** channel parameters as the Stage A checkpoint they consume; mismatch is a config error and should be asserted at startup. |
| `--background-K` | inherited (group size − 1; e.g. `0` for ITIT, `7` for GTGT-FM with K=8 images) | Existing OTA-NGT semantics; the K-hot label vector for GTGT-FM uses `K = background_K + 1`. |

The full sweep is `(2 algorithms) × (2 priv-loss forms) × (6 λ values) = 24 runs`, plus one Stage-C eval per run. Recommend running `λ = 0` baseline first to check that the pipeline reproduces existing OTA-NGT numbers.

---

## 6. Metrics

**Utility** (per epoch, on validation set):
- Firearm Acc@1
- ROC-AUC
- Recall on firearm class (system-critical false-negative rate)

(Already implemented in `main.py validate()`; reuse.)

**Privacy** (Stage C, on validation set):
- ITIT: post-hoc adversary's top-1 ImageNet accuracy.
- GTGT-FM: post-hoc adversary's mean per-class ROC-AUC + mean Average Precision.

**Trade-off plot** (paper figure):
- x-axis: privacy (lower = better, e.g., post-hoc adversary's top-1 accuracy).
- y-axis: utility (higher = better, e.g., firearm AUC).
- One marker per (algorithm, priv-loss, λ) configuration. Pareto frontier highlighted.

---

## 7. Implementation surface

New files:

- `privacy/__init__.py`
- `privacy/adversary.py` — `class AdversaryHead(nn.Module)` wrapping `layer3 + layer4 + avgpool + fc(N)`. Constructor takes `arch_name` and reuses `resnet_design2` building blocks so adversary and receiver are byte-identical except for the FC head.
- `privacy/losses.py` — `priv_loss_ce(adv_logits, label)`, `priv_loss_entropy(adv_logits)`, `priv_loss_entropy_multilabel(adv_logits)`. Selected via string config.
- `privacy/train_privacy.py` — entry point. Args: `--stage-a-ckpt`, `--priv-loss`, `--lambda`, `--k-adv`, `--stage-b-epochs`, `--recovery-epochs`, plus all existing OTA-NGT args (data, GT-alg, SNR, phase, etc.). Orchestrates Stage B → A_recovery and writes `stage_b_final.pth.tar`.
- `privacy/eval_privacy.py` — Stage C. Args: `--stage-b-ckpt`, `--stage-c-epochs`. Trains fresh adversary, dumps leakage metrics to JSON.

Modifications to existing files:

- `resnet_design2/my_resnet.py`: add an optional kwarg to `ResNet_GT._forward_impl` (and the `_phase` variant) that returns the post-channel feature tensor in addition to `(output, mean_x, power_x)`. Keeps existing call sites working.
- No modifications to `main.py`. Privacy training is a separate entry point so the original Stage A pipeline stays untouched.

Reuse:

- `data/GroupTestingDataset` (built; verified end-to-end by smoke test).
- `.venv` (torch + torchvision + tensorboard + sklearn).
- Stage A baseline checkpoint (the smoke-test checkpoint serves for development; full pretraining run for final results).

---

## 8. Open questions to resolve at implementation time

1. **Adversary optimizer**: SGD-momentum vs Adam. Adam tracks faster, but gives the adversary a learning-rate advantage (intentional under TTUR but worth verifying empirically).
2. **Adversary BatchNorm statistics**: whether `A`'s BN is in train mode during the encoder outer step (uses current batch stats, signal more variable) or eval mode (uses running stats, more stable but lags). Default: train mode.
3. **λ schedule**: constant per run vs warm-up over the first few epochs. Constant is the baseline; warm-up may help when `λ` is large.
4. **Stage A_recovery length**: how many epochs of `R` unfreeze before we lose more privacy than we recover utility. Empirical; default 2, sweep.
5. **GTGT-FM post-channel adversary input shape**: identical to ITIT after the merge, but verify the BN running stats transfer correctly when both algorithms share the encoder backbone.

---

## 9. Out of scope (for this spec)

- Differential-privacy noise budgets / formal DP guarantees.
- Defenses against an adversary that has *additional* side information (e.g., paired plaintext-feature samples).
- Multi-class group testing beyond firearm-vs-background.
- Other backbone architectures (ResNet-{18,50,152}). Same code should work via the existing `--arch` flag, but only ResNeXt-101 is in scope for the paper claim.
