# Third-party code

nanoRAE is released under the MIT License ([LICENSE](LICENSE)). It builds on the
[official RAE implementation](https://github.com/bytetriper/RAE) (MIT, Copyright (c) 2025 Boyang Zheng; commit `a4d18c4`),
which itself contains code from the projects below. Those parts keep their own licenses; the full texts are in [licenses/](licenses/).

**Two parts are for non-commercial use only:** the RAEv2-derived code (CC BY-NC 4.0) and `src/disc/diffaug.py`
(NVIDIA License, research or evaluation only).

| Files | Source | License |
|---|---|---|
| x-prediction (`prediction: 'x'`) in `src/stage2/transport/transport.py` and `src/stage2/transport/__init__.py`; the base head and internal guidance (`base_model_depth`, `forward_with_internalguidance`) in `src/stage2/models/DDT.py`, and the `'internal'` guidance option in `src/sample.py` and `src/sample_ddp.py` | [RAEv2](https://github.com/nanovisionx/RAEv2) (commit `8a0d238`), Copyright (c) 2025-2026 Jaskirat Singh, Boyang Zheng, Zongze Wu, Richard Zhang, Eli Shechtman, Saining Xie. Re-implemented in the code structure of RAE; the loss, the output conversion and the base head follow the official code | **CC BY-NC 4.0** ([licenses/LICENSE-RAEv2.txt](licenses/LICENSE-RAEv2.txt)) |
| `src/disc/diffaug.py` | [StyleGAN-T](https://github.com/autonomousvision/stylegan-t) (commit `36ab80c`), Copyright (c) 2023 NVIDIA Corporation & affiliates; included by RAE, unmodified | **NVIDIA License**, non-commercial ([licenses/LICENSE-StyleGAN-T.txt](licenses/LICENSE-StyleGAN-T.txt)) |
| `src/stage2/transport/` (paths, integrators, sampler), `src/train.py`, `src/sample.py`, `src/sample_ddp.py` | [SiT](https://github.com/willisma/SiT), Copyright (c) Meta Platforms, Inc. and affiliates; via RAE, modified | MIT ([licenses/LICENSE-SiT.txt](licenses/LICENSE-SiT.txt)) |
| `src/train_stage1.py` | Meta Platforms (file header); via RAE, modified | MIT |
| `src/stage2/models/lightningDiT.py` | [LightningDiT](https://github.com/hustvl/LightningDiT), Copyright (c) 2024 HUST Vision Lab; via RAE, modified | MIT ([licenses/LICENSE-LightningDiT.txt](licenses/LICENSE-LightningDiT.txt)) |
| `src/stage2/models/DDT.py` (DiT<sup>DH</sup>) | adapted by RAE from [DDT](https://github.com/MCG-NJU/DDT), which states no license | – |
| `src/disc/gan_loss.py`, `src/disc/lpips_utils.py`, `src/disc/lpips.py` | [taming-transformers](https://github.com/CompVis/taming-transformers), Copyright (c) 2020 Patrick Esser, Robin Rombach, Björn Ommer; via RAE | MIT ([licenses/LICENSE-taming-transformers.txt](licenses/LICENSE-taming-transformers.txt)) |
| `src/disc/lpips.py` (and the LPIPS weights it downloads) | [LPIPS](https://github.com/richzhang/PerceptualSimilarity), Copyright (c) 2018 Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang | BSD-2-Clause ([licenses/LICENSE-LPIPS.txt](licenses/LICENSE-LPIPS.txt)) |
| `src/stage1/decoders/decoder.py` | [Hugging Face Transformers](https://github.com/huggingface/transformers) (ViT-MAE decoder), Copyright 2022 Facebook AI and The HuggingFace Inc. team; via RAE, modified | Apache-2.0 ([licenses/LICENSE-Apache-2.0.txt](licenses/LICENSE-Apache-2.0.txt)) |

`src/stage2/transport/spherical.py` implements Algorithms 1-2 of [Learning on the Manifold (RJF)](https://arxiv.org/abs/2602.10099)
from the paper; no code was copied.
