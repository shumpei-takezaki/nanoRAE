# Changes from upstream

What this repository changes in the [official RAE implementation](https://github.com/bytetriper/RAE) (commit `a4d18c4`).
Code from other projects and its licenses: [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

- `src/stage1/encoders/cifar_dino.py`: the nanoDINO encoder (`CifarDinoViT`).
- `src/disc/dinodisc.py`: a 32px discriminator recipe (`recipe: 'cifar'`) whose frozen backbone is a DINO ViT-Tiny/4 from nanoDINO.
- `src/train_stage1.py`: flip-only augmentation at 32px; optional gradient checkpointing of the decoder.
- `src/stage2/models/lightningDiT.py`: `forward_with_cfg` takes `cfg_interval=(t_min, t_max)` like DDT (CFG and AutoGuidance sampling crashed otherwise).
- `src/stage2/transport/spherical.py`: RJF (`path_type: 'Spherical'`): per-token sphere, slerp path, tangent velocity, Jacobi weight
  (`jacobi_weight: false` gives RFM), exponential-map Euler sampler with output radius `sphere_radius`.
- `src/stage2/transport/`, `src/stage2/models/DDT.py`: x-prediction (`prediction: 'x'`) and a base head with internal guidance
  (`base_model_depth`, `guidance.method: 'internal'`), following RAEv2 (experiments in progress).
- `src/train.py`, `src/sample.py`: 32px / 10-class support; resuming from `ep-last.pt` (`src/utils/resume_utils.py`);
  `src/sample.py --per-class N --out <file>` for a grid with one class per row.
- `pyproject.toml` / `uv.lock`: environment managed with uv instead of conda.
- `scripts/`: CIFAR-10 data preparation, launchers for training and evaluation, IS, memorization and the analyses of the experiments.

Removed (not used for CIFAR-10): the ImageNet configs, the single-image demo (`src/stage1_sample.py`, `assets/`), `pack_images.py`,
and the conda / pip environment files. The upstream README is at [github.com/bytetriper/RAE](https://github.com/bytetriper/RAE/blob/main/README.md).
