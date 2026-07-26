# Experiment log — privacy-preserving OTA-NGT

A running, cross-cutting summary of **what we have run** and **what it tells us so far**, above the per-campaign detail docs. Newest synthesis at the bottom (§ "What we know now").

---

## 0. The system and the question

**OTA-NGT** splits a ResNet across a noisy channel: an **encoder `E`** (`conv1..layer2`) transmits features; a **receiver `R`** (`layer3..fc`) performs the useful task — **binary firearm-vs-background detection**. An eavesdropper taps the *same* transmitted features and trains a probe to recover the **fine-grained ImageNet class** (the private attribute).

**The goal:** make the transmitted representation *useful for the firearm task* but *uninformative about class identity*.

**How privacy is measured — "Stage C".** We freeze `E`, attach a fresh **adversary head**, train it to predict the original ImageNet class from the transmitted features, and report **`top1_imagenet_acc`** = fraction of background-val images whose true class the adversary recovers. Higher = more leakage. Two attacker strengths matter:
- **Kaiming adversary** (`--adv-init kaiming`): random-init, from-scratch — a *weak* attacker; a lower bound on leakage.
- **Worst-case adversary** (`--adv-init pretrained`): ImageNet-pretrained init — the *honest* attacker to trust for privacy claims.

**Two "utility costs" to watch:** firearm **recall** (of 50 firearm-val images) *and* the background **false-positive rate** (FPR over 48,800 background-val images). A detector can regain recall by simply firing more often (higher FPR), so read them together.

**v1 scope:** HSIC penalty only, ITIT (`--GT-alg 1 --background-K 0`), **σ=0** (no channel noise). VIB and σ>0 are v2.

---

## 1. Campaign A — adversarial/entropy privacy (2026-04-18) — **FAILED the refresh**

- **Detail doc:** `docs/superpowers/results/2026-04-18-itit-resnet18-no-noise-results.md`
- **What we ran:** Stage B fine-tunes `E` to **maximize the adversary's entropy** (`--priv-loss entropy`), i.e. *obfuscate* the class. ResNet-18, 1×L40, λ ∈ {0.1, 0.3, 1.0}. Leakage measured with a from-scratch adversary over the 979-way reorganized label set (chance ≈ 0.1%).
- **Result:** baseline leakage **35%**; best point (λ=1.0) dropped it to 29% at a **−14 pp utility** cost — but after a **2-epoch utility refresh, leakage rebounded fully to 35%** (0 pp privacy) while utility recovered.
- **What it told us:** the "privacy" was **obfuscation the receiver relearns** — adversarial scrambling does not remove class information, it hides it from *one* adversary, and recalibrating utility resurrects it. This motivated replacing the mechanism with a statistical-independence penalty (HSIC).

---

## 2. Campaign B — HSIC invariance v1, coarse sweep (2026-07-25)

- **Detail doc:** `docs/superpowers/results/2026-07-25-invariance-hsic-resnet18-results.md` (§1–§8)
- **What we ran:** replaced the adversary with an **HSIC independence penalty** — `E`+`R` trained *jointly* so transmitted features are statistically independent of class. ResNet-18, A40, coarse λ_H ∈ **{0, 1, 10, 100, 1000}** (0 = mandatory control: joint training + PK sampler, no HSIC). Utility (48k), Kaiming Stage C for all; worst-case (pretrained-adv) Stage C on **3 checkpoints only** (baseline, λ=100, refreshed λ=100). Refresh test on λ=100.
- **Results:**
  - Kaiming leakage: baseline **1.81%**, λ=100 **0.06%** (recall 38/50), λ=1000 0% (recall 0/50).
  - Refresh at λ=100: Kaiming leakage stayed ~**0.07%** post-refresh while recall recovered 38→49/50 — **privacy survived the refresh** (the decisive contrast with Campaign A, which rebounded 100%).
  - Worst-case (the sober view): baseline **27.80%**, λ=100 **3.07%** unrefreshed → **5.89%** refreshed (still ~4.7× below baseline).
- **What it told us:** HSIC is **qualitatively better** than the adversarial predecessor — it *removes* class-correlated structure rather than scrambling it, so a from-scratch attacker stays near-blind even after a refresh. **But** two gaps kept the verdict soft: the grid **skipped the entire 10→100 knee**, and the honest worst-case attacker (which reads ~15× higher than Kaiming) was measured on only 3 points.

---

## 3. Campaign C — V1 refinement, full frontier (2026-07-26, this session)

- **Detail doc:** same file, **§9**.
- **What we ran:** closed both gaps. Fine grid λ_H ∈ **{20, 30, 50, 70}** (train → utility → Kaiming Stage C) **and pretrained-adv (worst-case) Stage C across the WHOLE sweep** (baseline + all of {0,1,10,20,30,50,70,100,1000}). Picked **BEST = λ_H 30** and ran the refresh durability test incl. worst-case on the refreshed checkpoint. (Also rebuilt the sync-excluded dataset from image-net.org — see memory `imagenet-rebuild-gotchas`.)
- **Results (worst-case leakage vs firearm recall):**

  | λ_H | recall | FPR | **worst-case leak** |
  |---|---|---|---|
  | baseline | 50/50 | 2.16% | **27.80%** |
  | 10 | 50/50 | 0.66% | **26.92%** |
  | 20 | 49/50 | 1.33% | **22.01%** |
  | **30 (BEST)** | 48/50 | 1.43% | **18.56%** |
  | 50 | 47/50 | 1.88% | **14.18%** |
  | 70 | 46/50 | 2.64% | **8.66%** |
  | 100 | 38/50 | 2.56% | **3.07%** |
  | 1000 | 0/50 | 0.00% | **0.00%** |

  - **Smooth monotonic frontier, no cliff/elbow** — ~3–5 pp worst-case leakage per additional missed firearm.
  - **Kaiming badly understates it** — up to ~18× (λ=20: 1.24% Kaiming vs 22.0% worst-case on the *same* checkpoint).
  - **BEST=30 refresh:** worst-case rebounds **18.56% → 22.27%** (only ~1.25× below baseline); recall recovers 48→50/50 but **FPR triples 1.43% → 4.02%**.
- **What it told us:** the clean "HSIC buys privacy" headline was an **artifact of the weak Kaiming attacker**. Against the honest attacker there is **no operating point that is simultaneously recall-preserving, worst-case-private, and refresh-durable** — the entanglement wall.

---

## 4. Campaign D — V2-E0: does channel noise open an SNR window? (2026-07-26)

- **Detail doc:** `docs/superpowers/results/2026-07-26-v2e0-snr-window-resnet18.md`
- **What we ran (gated, on frozen V1 checkpoints — no VIB/new training):** a premise check for V2's "noise is a privacy lever independent of capacity" thesis. **Step 1** static margin/collapse gate; **Step 2** confirm on the baseline at {0,−5,−10} dB — firearm utility with the *receiver* adapted through noise (recall@FPR≤2%) and worst-case leakage with the *adversary retrained* through noise; **Step 3** matched-FPR frontier overlay + noise+HSIC combination.
- **Result — NO window.** The static gate *looked* promising (class-ID collapses ~6 dB before firearm). But at **matched recall@FPR≤2%**, pure noise sits **on/outside** the σ=0 HSIC frontier: at 50/50 recall, HSIC λ=30 leaks **18.6%** vs noise −5 dB **20.0%** (HSIC marginally better); HSIC also wins at ~37/50. The knees **coincide at matched utility** — the V1 entanglement wall. (A Step-2 reading briefly looked like a window; it was an artifact of matching noise's recall@2%FPR against HSIC's *0.5-threshold* recall — the matched-FPR overlay dissolves it. One nuance: noise+HSIC *combined* is mildly complementary, unconfirmed.)
- **What it told us:** raw channel noise is **not** an independent lever on this entangled code — it moves along the same wall as HSIC. Also surfaced two evaluator bugs (main.py input-space vs privacy feature-space ITIT noise; a latent `device.index=None` crash in `eval_privacy` at σ>0). → **V2 re-scoped capacity-first (E2/ResNeXt-101); drop the noise-alone E1 arm; VIB/noise only in E3 combination under a frozen-noise-floor.**

## What we know now (cumulative synthesis)

1. **Mechanism matters: independence > obfuscation.** HSIC removes class-correlated structure; the adversarial predecessor merely hid it. Only HSIC survives a refresh at high λ. This is a real, defensible qualitative result.
2. **Attacker strength matters: always report the pretrained-adv worst case.** The Kaiming attacker understated leakage up to ~18× and manufactured an illusory "free-privacy elbow" at low λ.
3. **On ResNet-18 / ITIT / σ=0 there is an entanglement wall.** Class identity and the firearm decision share the small (128-ch layer-2) subspace, so worst-case leakage and firearm utility trade off ~proportionally. At recall-preserving λ (≤30) the reduction is only ~1.3–1.5×; the ~9× cut needs recall 38/50.
4. **The recall-preserving privacy is also not durable.** At BEST=30 a routine 2-epoch refresh rebounds worst-case leakage to within ~1.25× of the unprotected baseline (and restores recall only by ~3×-ing the FPR). Durable *and* large privacy exists only where recall craters (λ=100).
5. **Net:** within this arch, HSIC is an honest, tunable **frontier**, not free privacy. Whether the frontier can be *bent* is a capacity question, not a noise question → **V2**.

---

## Open questions → V2

- **Does representational capacity bend the frontier?** A wider transmitted code (`resnext101_32x8d`: layer-2 = 512 ch vs ResNet-18's 128) may let class and task separate — buying a given worst-case-leakage reduction at less recall cost, and holding it under refresh. *(Primary V2 experiment — see `HANDOFF-v1-refine.md` → Next steps.)*
- ~~Does channel noise (σ>0) help?~~ **Answered (Campaign D): no — noise-alone does not beat the HSIC frontier at matched utility; it's on the same wall.** Only the noise+HSIC *combination* showed a mild (unconfirmed) hint → test in E3, not E1.
- **Is the entanglement fundamental to firearm-vs-class,** or an artifact of the ITIT tap point / K=0? (Test K>0 coalitions, alternate tap layers.)

*Index of detail docs: specs in `docs/superpowers/specs/`, plans in `docs/superpowers/plans/`, per-campaign results in `docs/superpowers/results/`.*
