# 14. RJF + AutoGuidance

- **Status:** done
- **Paper:** RJF (Learning on the Manifold) reports its guided result with AutoGuidance (scale 1.6); RAE Appendix I

## Goal

Both RJF ([12](12_rjf.md)) and AutoGuidance ([09](09_autoguidance.md)) improved FID on their own. We check whether they add up, for the
best model (DiT<sup>DH</sup>-S with a 512 head) and for the plain LightningDiT-S/1, which with RJF is almost as good as the head-512 model.
For S/1 we also check RFM (RJF without the Jacobi weight, [13](13_rfm.md)) with AutoGuidance.

## Setup

- Strong models: the RJF models of [12](12_rjf.md) (head 512: FID 7.81 at R = 13.66 / 6.81 at R = 18; S/1: 8.22 / 6.97) and the RFM S/1 of
  [13](13_rfm.md) (8.28 / 7.09), not retrained.
- Guides must predict the same kind of velocity (tangent to the unit sphere), so they are trained with RJF too: a short DiT<sup>DH</sup>-Ti + RJF
  for the head-512 model and a short LightningDiT-Ti/1 + RJF for S/1, 80 epochs each (the guide epoch that worked in [09](09_autoguidance.md)).
  The RFM model is guided by a LightningDiT-Ti/1 trained the same way with RFM.
- Scales w = 1.25, 1.42, 1.5; output radius R = 13.66 and 18. 50k FID and IS.

Configs: `configs/stage2/training/CIFAR10/{DiTDH-Ti-RJF,LightningDiT-Ti-RJF,LightningDiT-Ti-RFM}-guide_dinov2-r0_decXL-gan.yaml`,
`configs/stage2/sampling/CIFAR10/{DiTDH-S-h512-RJF,LightningDiT-S-RJF,LightningDiT-S-RFM}-AG_dinov2-r0_decXL-gan.yaml`

## Results

50k samples, FID / IS.

| Strong model | Guidance | R = 13.66: w = 1.25 | w = 1.42 | w = 1.5 | R = 18: w = 1.25 | w = 1.42 | w = 1.5 |
|---|---|---|---|---|---|---|---|
| DiT<sup>DH</sup>-S head 512 + RJF | AG (DiT<sup>DH</sup>-Ti + RJF) | 6.75 / 9.46 | 6.39 / 9.54 | **6.31** / 9.57 | 6.43 / 9.68 | 6.39 / 9.75 | 6.45 / 9.75 |
| LightningDiT-S/1 + RJF | AG (LightningDiT-Ti + RJF) | 6.59 / 9.44 | 6.15 / 9.54 | **6.07** / 9.58 | 6.41 / 9.64 | 6.54 / 9.69 | 6.68 / 9.69 |
| LightningDiT-S/1 + RFM | AG (LightningDiT-Ti + RFM) | 6.68 / 9.33 | 6.22 / 9.50 | **6.11** / 9.55 | 6.49 / 9.60 | 6.51 / 9.67 | 6.62 / 9.67 |

Summary (best over w where guided):

| Model | No guidance, R = 13.66 | No guidance, R = 18 | AG, no RJF | **RJF + AG** |
|---|---|---|---|---|
| DiT<sup>DH</sup>-S head 512 | 7.94 (no RJF) / 7.81 (RJF) | 6.81 | 6.48 | **6.31** |
| LightningDiT-S/1 | 8.70 (no RJF) / 8.22 (RJF) | 6.97 | – | **6.07** |
| LightningDiT-S/1, RFM | 8.28 | 7.09 | – | 6.11 (RFM + AG) |

- **RJF and AutoGuidance add up only partly.** For the head-512 model, RJF + AG (6.31) is only a little better than AG alone (6.48).
- **With AutoGuidance, a larger R no longer helps** (head 512: 6.31 at R = 13.66 vs. 6.39 at R = 18; S/1: 6.07 vs. 6.41). Without
  guidance R = 18 was clearly better.
- **The Jacobi weight matters little with guidance either:** RFM + AG gives 6.11 vs. 6.07 for RJF + AG (S/1), with the same dependence on
  w and R.
- **The plain LightningDiT-S/1 with RJF + AutoGuidance is the best model so far (FID 6.07)**, without a DDT head and with fewer parameters
  than the head-512 model.
- One seed; best w and R chosen on the same FID, so these are slightly optimistic.
