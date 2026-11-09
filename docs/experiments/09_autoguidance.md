# 09. AutoGuidance

- **Status:** done
- **Paper:** RAE Appendix I, Table 8 (gFID 1.51 → 1.13 with AutoGuidance; CFG without an interval does not improve FID)

## Goal

CFG only makes FID worse on our latent ([05](05_stage2_model_size.md)), as RAE reports for CFG without a guidance interval. RAE's best results use
AutoGuidance instead: a weak model guides the strong one, x̂ = x̂_weak + w(x̂_strong − x̂_weak). We check whether it improves FID on CIFAR-10.

## Setup

- Strong model: DiT<sup>DH</sup>-S with a 512 head ([05](05_stage2_model_size.md), FID 7.94), not retrained.
- Weak model: a short DiT<sup>DH</sup>-Ti run with the same recipe, stopped at epoch 80, using the checkpoints of epochs 20, 40 and 80. RAE uses
  early checkpoints of its smallest model (DiT<sup>DH</sup>-S at epochs 14-20 of an 800-epoch schedule).
- Scales w = 1.25, 1.42, 1.5, 2.0 (RAE uses 1.42-1.5). 50k FID and IS.

Configs: `configs/stage2/training/CIFAR10/DiTDH-Ti-guide_dinov2-r0_decXL-gan.yaml`, `configs/stage2/sampling/CIFAR10/DiTDH-S-h512-AG_dinov2-r0_decXL-gan.yaml`

## Results

50k samples, FID against CIFAR-10 train / IS. No guidance: FID 7.94, IS 9.22.

| Guide (DiT<sup>DH</sup>-Ti) | w = 1.25 | w = 1.42 | w = 1.5 | w = 2.0 |
|---|---|---|---|---|
| Epoch 20 | 10.45 / 9.90 | 14.61 / 10.01 | 16.72 / 10.03 | 29.27 / 9.85 |
| Epoch 40 | 9.31 / 9.85 | 12.51 / 9.95 | 14.24 / 9.99 | 25.02 / 9.82 |
| Epoch 80 | 6.71 / 9.52 | **6.48** / 9.68 | 6.50 / 9.71 | 7.40 / 9.86 |

- **AutoGuidance improves FID** where CFG only made it worse ([05](05_stage2_model_size.md)): 7.94 → **6.48** with the epoch-80 guide
  and w = 1.42, the scale RAE uses for its best result. IS also rises (9.22 → 9.68), unlike with CFG where FID and IS moved in opposite directions.
- **The guide must not be too weak.** Guides from epochs 20 and 40 make FID worse at every scale; only the epoch-80 guide helps. RAE uses
  even earlier checkpoints relative to its schedule (epochs 14-20 of 800, about 2%; our epoch 80 is 4% of 2000). With our small dataset
  and model, epochs 20-40 may simply be too poor a model (their samples were not evaluated).
- Only three guide checkpoints were tried; later or differently sized guides may do better. One seed.
