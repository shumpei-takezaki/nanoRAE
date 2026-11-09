# Pretrained weights

All weights are on [huggingface.co/shumpei777/nanoRAE](https://huggingface.co/shumpei777/nanoRAE) (about 14 GB in total), organized into
`encoders/`, `stats/`, `stage1/` and `stage2/`. Download them into `pretrained_models/`, where the `*_pretrained.yaml` configs read them:

```bash
uv run hf download shumpei777/nanoRAE --local-dir pretrained_models                       # everything
uv run hf download shumpei777/nanoRAE --local-dir pretrained_models --include <file> ...  # only some files
```

Every Stage 2 model also needs `stage1/decoder_vitxl_gan.pt`, `encoders/encoder_dinov2_vit_tiny4.pt` and `stats/normalization_stats.pt`.
Releases v0.1 and v0.2 of this code expect an older flat layout: download those with `--revision v0.2`.

Only inference weights (EMA) are released; training checkpoints are not.

## Encoders and shared files

| File | Content | Size |
|---|---|---:|
| `encoders/encoder_dinov2_vit_tiny4.pt` | DINOv2 ViT-Tiny/4 encoder (EMA teacher) | 22 MB |
| `encoders/encoder_dinov2_vit_s4.pt` | DINOv2 ViT-S/4 encoder (experiment 04) | 85 MB |
| `stats/normalization_stats.pt` | per-channel, per-position latent mean / var of the Tiny encoder (train, with flips) | 0.1 MB |
| `stats/cifar10_train_fid_stats.npz` | FID reference: Inception mean / covariance of the 50k CIFAR-10 training images (torch-fidelity `inception-v3-compat`, 2048-d) | 34 MB |
| `stats/cifar10_test_fid_stats.npz` | the same for the 10k CIFAR-10 test images (not used here; for comparisons against test) | 34 MB |
| `encoders/disc_dino_vit_tiny4.pt` | DINO (v1) ViT-Tiny/4, frozen backbone of the GAN discriminator (only for training) | 22 MB |

The encoders can also be loaded directly with nanoDINO: `from cifar_dino.export import load_backbone`.

All FIDs in this repository are against `cifar10_train_fid_stats.npz` (the real test images score 3.15 against it). To use the same reference without computing it, put it where
the evaluation scripts read it before running `scripts/prepare_cifar10.py` (which then keeps it):

```bash
uv run hf download shumpei777/nanoRAE --local-dir pretrained_models --include stats/cifar10_train_fid_stats.npz
mkdir -p data/cifar10 && cp pretrained_models/stats/cifar10_train_fid_stats.npz data/cifar10/train_fid_stats.npz
```

## Stage 1 decoders

EMA weights; inference configs in `configs/stage1/pretrained/`. rFID on the CIFAR-10 test set.

| File | Encoder | Decoder | GAN | Noise aug. | rFID | Experiment | Size |
|---|---|---|---|---|---:|---|---:|
| `stage1/decoder_vitb.pt` | Tiny | ViT-B | no | yes | 5.51 | [01](experiments/01_stage1_decoder_capacity.md) | 0.3 GB |
| `stage1/decoder_vitb_gan.pt` | Tiny | ViT-B | yes | yes | 6.57 | [01](experiments/01_stage1_decoder_capacity.md) | 0.3 GB |
| `stage1/decoder_vitl.pt` | Tiny | ViT-L | no | yes | 5.15 | [01](experiments/01_stage1_decoder_capacity.md) | 1.2 GB |
| `stage1/decoder_vitl_gan.pt` | Tiny | ViT-L | yes | yes | 5.80 | [01](experiments/01_stage1_decoder_capacity.md) | 1.2 GB |
| `stage1/decoder_vitxl.pt` | Tiny | ViT-XL | no | yes | 5.25 | [01](experiments/01_stage1_decoder_capacity.md) | 1.7 GB |
| `stage1/decoder_vitxl_gan.pt` | Tiny | ViT-XL | yes | yes | 5.71 | [01](experiments/01_stage1_decoder_capacity.md), used by Stage 2 | 1.7 GB |
| `stage1/decoder_vitxl_gan_nonoise.pt` | Tiny | ViT-XL | yes | no | 4.22 | [08](experiments/08_decoder_noise.md) | 1.7 GB |
| `stage1/decoder_vitxl_s4.pt` | S | ViT-XL | no | yes | 3.97 | [04](experiments/04_encoder_capacity.md) | 1.7 GB |
| `stage1/decoder_vitxl_gan_s4.pt` | S | ViT-XL | yes | yes | 3.71 | [04](experiments/04_encoder_capacity.md) | 1.7 GB |

## Stage 2 models

EMA weights, 2000 epochs, Tiny encoder. FID on 50k samples against CIFAR-10 train, without guidance.
Sampling configs: `configs/stage2/sampling/CIFAR10/<name>_pretrained.yaml`.

| File | Model | Config `<name>` | FID | Experiment | Size |
|---|---|---|---:|---|---:|
| `stage2/ditdh_s_h512_ema.pt` | **DiT<sup>DH</sup>-S with a 512-wide head (default)** | `DiTDH-S-h512` | **7.94** | [05](experiments/05_stage2_model_size.md) | 0.2 GB |
| `stage2/lightningdit_s_ema.pt` | LightningDiT-S/1, no DDT head | `LightningDiT-S` | 8.70 | [06](experiments/06_ddt_head.md) | 0.1 GB |
| `stage2/ditdh_s_h512_noshift_ema.pt` | 512-wide head, no time shift | `DiTDH-S-h512-noshift` | 9.01 | [07](experiments/07_time_shift.md) | 0.2 GB |
| `stage2/ditdh_ti_ema.pt` | DiT<sup>DH</sup>-Ti (18.5M) | `DiTDH-Ti` | 9.64 | [05](experiments/05_stage2_model_size.md) | 0.1 GB |
| `stage2/ditdh_s_ema.pt` | DiT<sup>DH</sup>-S with the official 2048-wide head | `DiTDH-S` | 13.59 | [02](experiments/02_stage2_ditdh_s.md) | 0.8 GB |
| `stage2/rjf_ditdh_s_h512_ema.pt` | 512-wide head + RJF (R = 18) | `DiTDH-S-h512-RJF` | 6.81 | [12](experiments/12_rjf.md) | 0.2 GB |
| `stage2/rjf_lightningdit_s_ema.pt` | LightningDiT-S/1 + RJF (R = 18) | `LightningDiT-S-RJF` | 6.97 | [12](experiments/12_rjf.md) | 0.1 GB |
| `stage2/rjf_lightningdit_ti_ema.pt` | LightningDiT-Ti/1 + RJF (R = 18) | `LightningDiT-Ti-RJF` | 9.22 | [12](experiments/12_rjf.md) | 0.03 GB |
| `stage2/rfm_lightningdit_s_ema.pt` | LightningDiT-S/1 + RFM (R = 18) | `LightningDiT-S-RFM` | 7.09 | [13](experiments/13_rfm.md) | 0.1 GB |
| `stage2/rfm_lightningdit_ti_ema.pt` | LightningDiT-Ti/1 + RFM (R = 18) | `LightningDiT-Ti-RFM` | 9.41 | [13](experiments/13_rfm.md) | 0.03 GB |

R is the per-token radius the RJF / RFM samples are scaled to before decoding (13.66 is the mean token norm of the normalized latent;
set `transport.params.sphere_radius` to change it).
The config `DiTDH-S-h512-nonoise` decodes the default model with `stage1/decoder_vitxl_gan_nonoise.pt` (experiment 08, FID 10.45).

## AutoGuidance

The weak guides are short runs (80 epochs). Download the strong model (above) as well.

| Guide file | Guides | Config `<name>` | w | FID | Experiment | Size |
|---|---|---|---:|---:|---|---:|
| `stage2/guide_ditdh_ti_ema.pt` | DiT<sup>DH</sup>-S with a 512-wide head | `DiTDH-S-h512-AG` | 1.42 | 6.48 | [09](experiments/09_autoguidance.md) | 0.07 GB |
| `stage2/guide_ditdh_ti_rjf_ema.pt` | 512-wide head + RJF (R = 13.66) | `DiTDH-S-h512-RJF-AG` | 1.5 | 6.31 | [14](experiments/14_rjf_autoguidance.md) | 0.07 GB |
| `stage2/guide_lightningdit_ti_rjf_ema.pt` | **LightningDiT-S/1 + RJF (R = 13.66), best** | `LightningDiT-S-RJF-AG` | 1.5 | **6.07** | [14](experiments/14_rjf_autoguidance.md) | 0.03 GB |
| `stage2/guide_lightningdit_ti_rfm_ema.pt` | LightningDiT-S/1 + RFM (R = 13.66) | `LightningDiT-S-RFM-AG` | 1.5 | 6.11 | [14](experiments/14_rjf_autoguidance.md) | 0.03 GB |
