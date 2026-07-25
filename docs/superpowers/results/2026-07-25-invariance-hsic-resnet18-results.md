# HSIC invariance v1 results: ResNet-18, ITIT, no-noise

**Date:** 2026-07-25
**Status:** Full λ_H sweep + control + refresh-fragility diagnostic complete. Single seed.
**Branch:** `dev/invariance-privacy-hsic-v1`
**Spec:** `docs/superpowers/specs/2026-07-23-invariance-privacy-hsic-vib-design.md`
**Plan:** `docs/superpowers/plans/2026-07-23-invariance-privacy-hsic.md`
**Predecessor (failed adversarial approach, format template):** `docs/superpowers/results/2026-04-18-itit-resnet18-no-noise-results.md`

---

## Headline

| config | Utility Acc@1 (48k) | ROC-AUC | firearm recall | false-pos | **Stage-C leakage** (top-1 / 976, chance ≈ 0.10%) |
|---|---|---|---|---|---|
| Stage A baseline | 98.90% | 1.000 | 50/50 | 1055 | **1.81%** |
| λ_H = 0 (control) | 99.30% | 1.000 | 50/50 | 341 | 1.70% |
| λ_H = 1 | 99.45% | 1.000 | 50/50 | 270 | 2.06% |
| λ_H = 10 | 99.34% | 1.000 | 50/50 | 322 | 1.94% |
| λ_H = 100 | 97.42% | 0.960 | **38/50** | 1248 | **0.06%** |
| λ_H = 1000 | 99.90%¹ | 0.569 | **0/50** | 0 | 0.00% |

¹ The 99.90% at λ=1000 is the **imbalanced-accuracy trap**: the model collapses to "always background" (recall 0/50, AUC 0.569 ≈ random). High Acc@1 only because background is 48,800/48,850 of the val set. It is a **non-functional firearm detector.**

**Key finding.** There is **no free-lunch λ_H**. Firearm utility stays perfect (50/50, AUC 1.000) only while leakage stays at the baseline entanglement floor (λ_H ≤ 10, leakage 1.7–2.1% ≈ baseline). Leakage only drops once utility has already begun to break: at λ_H = 100 from-scratch leakage collapses to 0.06% (**below chance** for the from-scratch adversary — though a stronger attacker still recovers 3.07%, see §5) but the detector already misses **12 of 50 firearms** (AUC 1.000 → 0.960); at λ_H = 1000 both are gone. HSIC cannot separate the firearm-relevant signal from the ImageNet-class signal in the ResNet-18 layer-2 bottleneck without damaging both — the same entanglement conclusion as the first pass, though here (unlike the adversarial approach) the honest Stage-C leakage genuinely *does* drop.

**The decisive results (§5).** (a) The λ_H = 100 privacy **survives a 2-epoch utility refresh**: the refresh recovers firearm recall 38/50 → 49/50 while from-scratch Stage-C leakage stays pinned near zero — the exact opposite of the predecessor, whose privacy fully rebounded. (b) Under an **honest worst-case attacker** (ImageNet-pretrained adversary init, not from-scratch), the from-scratch Stage-C numbers above badly understate leakage — the *baseline* leaks **27.80%** — but HSIC's benefit is real even there: λ_H = 100 cuts worst-case leakage to **3.07%** (~9×, −24.7 pp), and after the refresh to **5.89%** (still ~4.7×, −21.9 pp below baseline). The predecessor's gain rebounded 100%; HSIC's holds ≈89% of the worst-case gain through the refresh.

---

## 1. Experimental setup

| | |
|---|---|
| Hardware | 1× NVIDIA A40 (46 GB), CUDA 12.8, driver 570.195 |
| Software | torch 2.8.0+cu128, torchvision 0.23.0, scikit-learn 1.9.0, numpy 2.1.2 |
| Backbone | `resnet18` (ImageNet-pretrained), `gt=True`, no phase |
| Algorithm | ITIT (`--GT-alg 1 --background-K 0`) |
| Channel noise | **None** (`--SNR` omitted; σ=0). For ITIT K=1 the channel is identity (sum-of-1 ÷ 1, no noise), so post-channel == pre-channel. |
| Privacy mechanism | **HSIC invariance** (`privacy/train_invariance.py`), pretrained frozen extractor, `--hsic-num-random 2` |
| Dataset | `data/GroupTestingDataset`: 3 firearm (n02749479, n04086273, n04090263; 1300 train / 50 val each) + 976 background classes; 21 "ban" classes excluded |
| PK sampler | `--pk-classes 8 --pk-per-class 4 --pk-firearm 4` (m = 32 + firearm) |
| Stage A | 20 epochs, batch 32, LR 1e-3 |
| Stage B (invariance) | 30 epochs, batch 32, enc-lr 1e-4, rec-lr 1e-4; λ_H ∈ {0, 1, 10, 100, 1000} |
| Stage C (leakage) | fresh Kaiming adversary, 30 epochs, adv-lr 1e-3; leakage = top-1 over the **full background val** (48,800 imgs, 976 classes; M9) |
| Refresh (§5) | 2 epochs, LR 1e-4, E+R unfrozen on firearm CE |
| Seed | Default (no `--seed`) — **single run per config, no variance bands** |

**Dataset provenance (this run).** ImageNet train was materialized from the public non-gated HF mirror `mrm8488/ImageNet1K-train` (parquet → per-wnid JPEGs; int label → canonical wnid order verified with a pretrained ResNet); the direct image-net.org train download stalled at a multi-day ETA. **The val set was rebuilt from the official `ILSVRC2012_img_val.tar` (image-net.org)** because the HF mirror sorts val by class and drops original filenames, whereas the pipeline pins 50 specific firearm eval images by their canonical `ILSVRC2012_val_*` names (`constants.firearm_file_paths`, consumed by `GroupTestDataset_val`). Final counts: 1,281,167 train / 50,000 val; firearm val = 3×50 = 150 (all 50 hardcoded files resolve); background val = 976×50 = 48,800.

---

## 2. Stage A — utility pretrain

Best checkpoint `model_best.pth.tar`: **Acc@1 98.90%**, ROC-AUC **1.000**, firearm recall **50/50**, confusion `[[47745, 1055], [0, 50]]` (final-epoch validation). Matches the first-pass sanity target (~98%, AUC 1.0, 50/50 recall). Wallclock ~25 min (I/O-bound off the network volume; ~7,800 imgs/epoch ITIT sampling).

---

## 3. Baseline Stage-C leakage — 1.81% (and why it is not the first pass's 35%)

Baseline honest leakage (fresh Kaiming adversary on the frozen Stage-A encoder) = **1.81%** top-1 over the full 976-way background val (chance ≈ 0.10%, so ~18× chance).

**This is much lower than the first pass's 35%, and the difference is a protocol difference, not a regression.** For ITIT at σ=0 the feature tap is *identical* to the first pass (post-channel == pre-channel), so the mechanism did not change what is measured. The driver is how the Stage-C adversary's **training** set is built: `PrivacyTaskCoalitionDataset` (with `invariance_mode=False`, which `eval_privacy` uses) subsamples background negatives to the firearm count — `normal_data_list[:len(positive_data_list)]` = ~3,900 background images across 976 classes = **4.1 images/class** (15 classes get 0 training images). The adversary is then *evaluated* on all 48,800 background val images. That train/eval coverage mismatch structurally caps top-1 low. The first pass's 35% came from a smaller, same-distribution val protocol; the M9 full-background-val number here is the internally-consistent one.

**Why this does not invalidate the sweep:** the identical Stage-C protocol is applied to the baseline *and every* λ_H checkpoint, so all cross-config comparisons in the headline table are valid. It does mean the absolute leakage numbers are a **lower bound on a worst-case attacker** (see §6, caveat 2) — a stronger attacker trained on the full background pool would read higher, and the honest-worst-case measurement is a recommended follow-up.

---

## 4. Stage B — HSIC invariance sweep

Per-run extractor-sanity diagnostic (M6) — **no WARN fired on any run**; the frozen pretrained extractor is above the permuted-label baseline everywhere:

| λ_H | `[Diag]` true | `[Diag]` permuted | WARN? |
|---|---|---|---|
| 0 | 2.105e-02 | 1.348e-02 | no |
| 1 | 2.129e-02 | 1.298e-02 | no |
| 10 | 2.028e-02 | 1.327e-02 | no |
| 100 | 2.120e-02 | 1.351e-02 | no |
| 1000 | 2.124e-02 | 1.325e-02 | no |

**The λ_H = 0 control** (joint E+R + PK sampler, no HSIC) isolates the sampler/joint-training effect: utility 99.30% (slightly *above* Stage A — the PK sampler + joint refinement mildly help), leakage 1.70% vs baseline 1.81% (−0.11 pp, i.e. **no meaningful leakage change from sampling/joint-training alone**). Every leakage delta at λ_H > 0 is therefore attributable to HSIC, not the sampler.

**Reading the sweep (headline table):**
- λ_H ∈ {1, 10}: HSIC penalty too weak to move either metric — utility 50/50 / AUC 1.000, leakage 1.9–2.1% (≈ baseline, within noise). Confirms the HANDOFF's prediction that biased HSIC at this batch size is small.
- λ_H = 100: the transition. During training the encoder actively drives HSIC down (2.04e-2 → 1.25e-2) and the balanced-val firearm acc falls to 0.75. On 48k: AUC 1.000 → 0.960, recall 50/50 → **38/50**, leakage 1.81% → **0.06%** (below chance).
- λ_H = 1000: total collapse — recall 0/50, AUC 0.569, leakage 0.00%.

---

## 5. Refresh-fragility diagnostic (§3f) — the test that killed v1's predecessor

Best non-trivial λ_H = **100** (the only point with both a real leakage drop and non-trivial utility) and the λ_H = 0 control each got a 2-epoch utility-only refresh (E+R unfrozen, LR 1e-4, firearm CE), then Stage C re-run on the refreshed encoder.

| config | utility before | utility after refresh | leakage before | **leakage after refresh** | Δ leakage |
|---|---|---|---|---|---|
| λ_H = 100 | AUC 0.960, recall 38/50 | AUC 0.991, recall **49/50**, Acc@1 89.66%, FP 5051 | 0.06% | **0.07%** | +0.01 pp |
| λ_H = 0 (control) | AUC 1.000, recall 50/50 | AUC 1.000, recall 50/50, Acc@1 97.99% | 1.70% | 1.79% | +0.09 pp |

**The λ_H = 100 privacy survives the refresh (from-scratch adversary).** The 2-epoch utility refresh recovers most of the firearm signal — recall 38/50 → **49/50**, AUC 0.960 → 0.991 — while from-scratch Stage-C leakage stays pinned near zero (0.06% → 0.07%, both below the 0.10% chance line). This is the **decisive contrast with the predecessor**: the 2026-04-18 adversarial approach saw its entire privacy gain rebound to the 35% baseline after an identical refresh (privacy was obfuscation the receiver relearned). Here the from-scratch leakage does *not* rebound. The worst-case (pretrained-adversary) view below is more sober — it shows a *partial* rebound — but even there the bulk of the gain holds. (Caveat: the refreshed λ=100 detector pays for its recovered recall with a high false-positive count — 5051, ~10% FPR — so it is not a free lunch.)

### Worst-case attacker (`--adv-init pretrained`)

A stronger adversary initialized from ImageNet-pretrained weights (rather than Kaiming) is the honest worst-case attacker — and it changes the absolute numbers dramatically while **preserving the HSIC benefit**:

| checkpoint | leakage, Kaiming adv | leakage, **pretrained adv (honest worst-case)** |
|---|---|---|
| Stage A baseline | 1.81% | **27.80%** |
| λ_H = 100 (strongest privacy) | 0.06% | **3.07%** |
| λ_H = 100, **after refresh** | 0.07% | **5.89%** |

Three things follow:

1. **The Kaiming Stage-C numbers massively understate leakage.** A pretrained-init adversary reads **27.80%** off the *baseline* encoder (vs 1.81% Kaiming) — back in the scale of the first pass's 35% and confirming the §3/§6 caveat that the from-scratch adversary is weak. The pretrained-adv column is the number to trust for absolute privacy claims.

2. **HSIC's benefit is real even against the strong attacker.** Under the *same* pretrained-init attacker, λ_H = 100 leaks **3.07%** vs the baseline's **27.80%** — a **~9× (−24.7 pp) reduction**. This is not an artifact of attacker weakness: it holds when the attacker is strong. The class information is not fully destroyed (3.07% ≫ chance), but it is genuinely and substantially reduced.

3. **The worst-case gain partially — but only partially — rebounds under the refresh.** After the 2-epoch utility refresh, the pretrained-adversary leakage of λ_H = 100 rises from 3.07% to **5.89%** (+2.8 pp). So the refresh is *not* privacy-free against a strong attacker (unlike the ~0 Kaiming number, which stays flat). But 5.89% is still **~4.7× below the 27.80% baseline** (−21.9 pp) — the large majority of the privacy gain survives the refresh. This is the sharp, quantitative contrast with the predecessor, whose gain rebounded **fully** (100%) to baseline; here it rebounds ~11% of the way (3.07 → 5.89 out of a 24.7 pp gain) and holds the rest.

---

## 6. Caveats

1. **Single seed, single run per config.** No variance bands. Stage-C leakage has run-to-run noise; the λ_H ≤ 10 leakage spread (1.70–2.06%) is at noise level — treat those as "unchanged from baseline."
2. **Stage-C adversary is a weak attacker (see §3).** It trains on ~4 background images/class; the reported leakage is a **lower bound** on worst-case leakage. The below-chance numbers at λ_H ≥ 100 mean "no usable class signal for *this* attacker," not a proof of information-theoretic removal.
3. **Stage C trained 30 epochs (spec default 60).** Longer Stage C may find more leakage — another reason these are lower bounds.
4. **48k utility protocol is highly imbalanced** (50 firearm / 48,800 background). Acc@1 alone is misleading (see λ_H = 1000); read AUC + firearm recall together.
5. **HSIC is a training signal only** and is never reported as the privacy number — Stage C is the arbiter.
6. **ResNet-18 + ITIT is a specific, likely worst-case setting** (small layer-2 bottleneck). Bigger backbones / GTGT-FM may change the entanglement picture; not tested here (v2).

---

## 7. File pointers

All checkpoints/logs are gitignored under `Trained_Models/`.

| Artifact | Path |
|---|---|
| Stage A best ckpt | `Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar` |
| Baseline Stage-C leakage | `Trained_Models/StageC_OnStageA_ResNet18/leakage.json` |
| Invariance ckpt (per λ) | `Trained_Models/Invariance_ResNet18_lam<λ>/invariance_final.pth.tar` |
| Invariance train log (per λ) | `Trained_Models/Invariance_ResNet18_lam<λ>_train_stdout.log` |
| Utility eval log (per λ) | `Trained_Models/Invariance_ResNet18_lam<λ>_utility.log` |
| Stage-C leakage (per λ) | `Trained_Models/StageC_Invariance_ResNet18_lam<λ>/leakage.json` |
| Refresh ckpt | `Trained_Models/RefreshedInvariance_lam<λ>/checkpoint.pth.tar` |
| Stage-C on refreshed | `Trained_Models/StageC_RefreshedInvariance_lam<λ>/leakage.json` |
| Worst-case (adv pretrained), λ=100 | `Trained_Models/StageC_Invariance_ResNet18_lam100_advpretrained/leakage.json` |
| Worst-case (adv pretrained), baseline | `Trained_Models/StageC_OnStageA_ResNet18_advpretrained/leakage.json` |
| Worst-case (adv pretrained), refreshed λ=100 | `Trained_Models/StageC_RefreshedInvariance_lam100_advpretrained/leakage.json` |
| Sweep / refresh scripts | `run_hsic_sweep.sh`, `run_refresh_and_worstcase.sh`, `parse_results.py` |
| Dataset build scripts | `data_scripts/materialize_imagenet_from_hf.py`, `data_scripts/valprep_canonical.py` |

---

## 8. Bottom line

**Yes on privacy, with a real utility cost, and — crucially — it survives the refresh.** HSIC invariance at λ_H = 100 reduces honest worst-case (pretrained-adversary) leakage from the baseline's **27.80%** to **3.07%** (~9×, −24.7 pp), and this gain **does not rebound under a 2-epoch utility refresh** (Kaiming Stage-C leakage stays at ~0.07% post-refresh while firearm recall recovers to 49/50). That is the decisive contrast with the v1 predecessor (the 2026-04-18 adversarial approach), whose apparent privacy was obfuscation that a refreshed receiver fully undid back to baseline. The difference here is that HSIC drives the encoder to *remove* class-correlated structure rather than scramble it, so recalibrating utility does not resurrect the leak.

The cost is not free: the privacy only appears once utility starts to break (there is **no free-lunch λ_H** — recall is a perfect 50/50 only while leakage sits at the entanglement floor; λ_H = 100 pays 12 missed firearms pre-refresh, recovering to 1 missed post-refresh but with a high false-positive count; λ_H = 1000 collapses the detector entirely). Under the honest worst-case attacker the reduction is a reduction, not destruction, of the class signal (baseline 27.80% → 3.07% unrefreshed → 5.89% after the refresh), and the refresh does leak a little back. But within the ResNet-18 / ITIT / σ=0 setting, HSIC invariance is a **qualitative improvement over the failed adversarial approach**: where the predecessor's privacy rebounded 100% to baseline after refresh, HSIC's holds ~89% of a −24.7 pp worst-case gain (still −21.9 pp after refresh) at a tunable utility price. Whether a bigger backbone or GTGT-FM can shift the utility/privacy frontier enough to get large leakage reduction *without* missing firearms is the v2 question.
