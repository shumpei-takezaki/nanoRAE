# Training from scratch

The configs read the frozen encoder and the frozen backbone of the GAN discriminator (a DINO (v1) ViT-Tiny/4) from
`pretrained_models/` (see [weights.md](weights.md)); both were trained with [nanoDINO](https://github.com/shumpei-takezaki/nanoDINO).
All runs use a single GPU and bf16. Each launcher resumes automatically; add `--wandb` to log to Weights & Biases
(`WANDB_KEY`, `ENTITY`, `PROJECT` in `.env`).

Prepare the data first (see the [README](../README.md#-quick-start)): `uv run python scripts/prepare_cifar10.py`.
It computes the FID reference statistics on the GPU unless they are already there; the released ones are on Hugging Face
(see [weights.md](weights.md#encoders-and-shared-files)).

```bash
# Stage 1: decoder without GAN (400 epochs), then GAN fine-tuning (100 epochs)
scripts/train_stage1.sh configs/stage1/training/cifar_dinov2_decXL_nogan.yaml dinov2_decXL_nogan
scripts/train_stage1.sh configs/stage1/training/cifar_dinov2_decXL_gan_ft.yaml dinov2_decXL_gan_ft
PYTHONPATH=src uv run python src/extract_decoder.py --config configs/stage1/training/cifar_dinov2_decXL_gan_ft.yaml \
  --ckpt ckpts/stage1/dinov2_decXL_gan_ft/checkpoints/ep-last.pt --use-ema --out ckpts/stage1/dinov2_decXL_gan_ft/decoder_ema.pt
scripts/eval_stage1.sh configs/stage1/pretrained/cifar_dinov2_decXL_gan_ft.yaml dinov2_decXL_gan_ft test   # PSNR / SSIM / LPIPS / rFID

# Latent normalization statistics -> stats/
uv run torchrun --standalone --nproc_per_node=1 src/calculate_stat.py --config configs/stage1/pretrained/cifar_dinov2_decXL_gan_ft.yaml \
  --data-path data/cifar10/train --sample-dir stats --image-size 32 --per-proc-batch-size 500 --num-workers 4 --momentum 0.02

# Stage 2: DiT^DH-S with a 512-wide head (2000 epochs), then FID on 50k samples against CIFAR-10 train
scripts/train_stage2.sh configs/stage2/training/CIFAR10/DiTDH-S-h512_dinov2-r0_decXL-gan.yaml ditdh_s_h512_decXLgan
scripts/eval_stage2.sh configs/stage2/sampling/CIFAR10/DiTDH-S-h512_dinov2-r0_decXL-gan.yaml ditdh_s_h512_decXLgan_last_cfg1.0 1.0
```

The inference configs in `configs/stage1/pretrained/` read the released decoders from `pretrained_models/`; to evaluate a decoder
you trained yourself, point `pretrained_decoder_path` to its `decoder_ema.pt`.
Configs for the other decoders (B / L, with and without GAN) are in `configs/stage1/`. The commands for each experiment
are at the end of its file in [experiments/](README.md).

## Training time

Wall-clock times on one A100-40GB (bf16 unless noted):

| Run | Time |
|---|---|
| Encoder, nanoDINO DINOv2 ViT-Tiny/4 / ViT-S/4, 1000 epochs | ~16.5 h (fp32) / ~18 h (TF32) |
| Stage 1 decoder without GAN, 400 epochs: B / L / XL | ~21 h (fp32) / ~10.5 h / ~12.5 h |
| Stage 1 GAN fine-tuning, 100 epochs: B / L / XL | ~6 h (fp32) / ~4 h / ~5 h |
| Stage 2, 2000 epochs: DiT<sup>DH</sup>-S with the 2048 head / with the 512 head / DiT<sup>DH</sup>-Ti | ~31 h / ~13 h / ~9 h |
| FID on 50k samples: without / with CFG | ~20 min / ~35 min |
