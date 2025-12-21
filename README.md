# Mask-CDKD

PyTorch implementation of the **model definitions** for:

**Mask-CDKD: A Source-Free and Label-Free Cross-Domain Knowledge Distillation Framework from SAM for Satellite Onboard VHR Land-Cover Mapping**  
(ISPRS Journal of Photogrammetry and Remote Sensing)

This repository is **model-only**: it contains the core network modules used in our paper (teacher/student architectures and the distiller wiring). It is intended for **inspection, integration, and minimal forward tests**.

> **Pretrained weights: coming soon.**

## Installation

```bash
conda create -n maskcdkd python=3.10 -y
conda activate maskcdkd
pip install -r requirements.txt
````

> Install a compatible PyTorch build for your CUDA version from the official PyTorch website.

## Quick sanity check (forward only)

This is a minimal forward test using random input. It does **not** reproduce paper results.

```python
import torch
from maskcdkd.distiller import ClassificationDistiller

B = 2
images = torch.randn(B, 3, 1024, 1024)

# 1024 / 16 = 64 => 64*64 = 4096 patch tokens
mask_s = torch.zeros(B, 4096, dtype=torch.bool)
mask_t = torch.zeros(B, 4096, dtype=torch.bool)
mask_diff = torch.zeros(B, 4096, dtype=torch.bool)

distiller = ClassificationDistiller(sd=False, teacher_pretrained=None)

pred_s, pred_t, mae_s, mae_t = distiller(images, mask_s, mask_t, mask_diff)
print(len(pred_s), mae_s.shape)
```

## Dataset (LuoJiaCDKD-100K)

LuoJiaCDKD-100K contains **100,801 unlabeled 1024×1024** VHR remote sensing image tiles curated from multiple sources.

* Download: **(add link here)**
* Please comply with the original licenses/terms of any upstream datasets if you redistribute derived subsets.

## SAM checkpoints

We use **SAM** as a teacher and do **not** redistribute SAM weights. Please obtain checkpoints from the official SAM repository and follow its license.

## License

* Code license: see `LICENSE`
* Third-party components: this repository may include/adapt code from upstream projects. Please keep original copyright headers and ensure license compatibility.

## Contact

For questions or issues, please open a GitHub Issue or contact the first author or the corresponding author.