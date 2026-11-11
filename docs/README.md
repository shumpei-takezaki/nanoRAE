# Experiments

Reproductions of findings from RAE-related papers on CIFAR-10. One file per experiment in `experiments/`, figures and numbers
in `figures/<experiment>/`. Unless noted, the encoder is the DINOv2 ViT-Tiny/4 from nanoDINO (no registers, frozen) and the
latent is `[192, 8, 8]`.

rFID compares the reconstructions of the CIFAR-10 test set (10k images) with the originals (the paper uses the
ImageNet validation set). "train rFID" is the same on the training set. FID compares 50k generated samples with CIFAR-10 train.

| # | Experiment | Paper | Main result |
|---|---|---|---|
| 01 | [Stage 1: decoder size and GAN](experiments/01_stage1_decoder_capacity.md) | RAE Table 1b | On train, larger decoders are better (XL + GAN: train rFID 0.89). On test, all decoders overfit and the paper's trend does not hold |
| 02 | [Stage 2: DiT<sup>DH</sup>-S generation](experiments/02_stage2_ditdh_s.md) | RAE Sec. 5 | FID 13.59 (50k, no CFG). Best at epoch 500; CFG 3.0 hurts |
| 03 | [Single-image overfitting](experiments/03_single_image_overfit.md) | RAE Sec. 4.1, Table 3 | For d < n the loss stops at the bound (n − d)/n; it converges for d ≥ n. Reproduced |
| 04 | [Encoder size (Tiny vs. S; B coming soon)](experiments/04_encoder_capacity.md) | RAE Table 1c | With S, reconstruction improves on every metric (rFID 5.25 → 3.97, PSNR +2.6 dB), unlike the paper (nearly constant) |
| 05 | [Stage 2 model size (smaller DiT<sup>DH</sup>)](experiments/05_stage2_model_size.md) | RAE Fig. 6, Table 17 | A 512-wide DDT head (≈ 2.7n) beats the official 2048 head: FID (50k) 7.94 vs. 13.59 with 4.5× fewer parameters; DiT<sup>DH</sup>-Ti (18.5M) 9.64. No model memorizes by the Gu et al. criterion; the Carlini score shows the official model makes ~10× more near-copies |
| 06 | [Is the DDT head needed?](experiments/06_ddt_head.md) | RAE Table 6, RJF | Removing the 512 head (plain LightningDiT-S/1) costs a little: FID 8.70 vs. 7.94. A plain DiT with width = n (Ti/1) diverged |
| 07 | [Ablation: noise-schedule shift](experiments/07_time_shift.md) | RAE Table 4 | The shift helps less than in the paper: FID 9.01 without vs. 7.94 with (α ≈ 1.73) |
| 08 | [Ablation: decoder noise augmentation](experiments/08_decoder_noise.md) | RAE Table 5 | Reproduces the trade-off: without noise, rFID improves (5.71 → 4.22) but gFID worsens (7.94 → 10.45) |
| 09 | [AutoGuidance](experiments/09_autoguidance.md) | RAE Appendix I, Table 8 | AutoGuidance helps where CFG did not: FID 7.94 → 6.48 (w = 1.42, guide = DiT<sup>DH</sup>-Ti at epoch 80); weaker guides (epochs 20, 40) hurt |
| 10 | [Latent geometry](experiments/10_latent_geometry.md) | RJF Fig. 2–3 | The latent lies on a sphere of radius √192; the straight path shrinks the norm to 71% midway |
| 11 | [Radial / angular parts of the loss](experiments/11_radial_angular.md) | RJF Fig. 4 | The angular loss alone goes down to 0.004 even at d = 64 (full stays above the bound); nearly trivial for a single image |
| 12 | [RJF: flow matching on the sphere](experiments/12_rjf.md) | RJF Tables 3-4 | RJF makes the plain DiT with width = n trainable (Ti/1: diverged → FID 11.54) and helps less for wider models (S/1 8.70 → 8.22, head 512 7.94 → 7.81). A larger output radius helps further (head 512: 6.81 at R = 18) |
| 13 | [RJF ablation: no Jacobi weight (RFM)](experiments/13_rfm.md) | RJF Table 3 | The geometry alone (RFM) already makes Ti/1 trainable (11.80); the Jacobi weight adds 0.1-0.3 FID, as in the paper |
| 14 | [RJF + AutoGuidance](experiments/14_rjf_autoguidance.md) | RJF, RAE Appendix I | Best so far: LightningDiT-S/1 + RJF + AutoGuidance, FID 6.07 (head 512: 6.31; RFM + AutoGuidance: 6.11). With guidance, a larger output radius R no longer helps |
| 15 | [RAEv2: x-prediction](experiments/15_x_prediction.md) | RAEv2 (Stage 2) | x-prediction alone does not help on this latent: FID 8.22 vs. 7.94 with velocity prediction |
| 16 | [RAEv2: base head and internal guidance (REPA Guidance)](experiments/16_internal_guidance.md) | RAEv2 Tables 16-17 | Best so far: FID 4.70 at w = 3.0 (AutoGuidance 6.48, RJF + AutoGuidance 6.07), with a single model. The base head alone hurts (9.66 without guidance). x-prediction beats velocity only with guidance (4.70 vs. 5.45), as in the paper's Table 17 |

Next steps: see the roadmap in the [README](../README.md).

See also: [pretrained weights](weights.md), [training from scratch](training.md), [changes from upstream](changes.md).

Papers: RAE [arXiv 2510.11690](https://arxiv.org/abs/2510.11690), RJF (Learning on the Manifold) [arXiv 2602.10099](https://arxiv.org/abs/2602.10099), RAEv2 [arXiv 2605.18324](https://arxiv.org/abs/2605.18324).
