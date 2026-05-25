# USE: A Unified Self-Ensembling Framework for Test-Time Prompt Tuning

[![Venue](https://img.shields.io/badge/Venue-ICML%202026-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

This is the official PyTorch implementation of the paper **"USE: A Unified Self-Ensembling Framework for Test-Time Prompt Tuning"**, accepted as a regular paper at **ICML 2026**.

## 💡 Abstract
Test-time adaptation (TTA) has emerged as a popular paradigm for improving the performance of vision–language models (e.g., CLIP) on downstream tasks.
Among existing CLIP-based TTA methods, Test-Time Prompt Tuning (TPT) is a pioneering work that optimizes textual prompts using multiple test-time augmentations and remains a strong baseline to date.
In this work, we revisit TPT and reveal that its optimization can be interpreted as implicitly learning from self-generated pseudo labels.
Building on this perspective, we propose a unified self-ensembling framework USE that ensures consistency between the optimization and inference stages.
During optimization, we introduce a simple yet effective self-ensembling SE strategy that emphasizes the test image itself over its augmented views adaptively to obtain more reliable pseudo labels.
To fully exploit the potential of augmentations, we further apply the same strategy at inference time, unifying the objectives of both stages.
Notably, SE can also act as a lightweight optimization-free TTA method.
Extensive experiments across multiple datasets demonstrate that SE and USE outperform their counterparts, respectively.
Furthermore, SE yields consistent performance gains when integrated with existing TTA methods.
> 🌟 **Note on Terminology "Unified":** As mentioned by the anonymous reviewer, to avoid any ambiguity, we unify the optimization and inference stages into a single framework, ensuring consistency between them.

## 📂 Data Preparation

Please refer to the dataset guidelines from the [TPT repository](https://github.com/azshue/TPT) to download the following datasets:

* **ImageNet & OOD Variants:** ImageNet-1K, ImageNet-A, ImageNet-V2, ImageNet-R, ImageNet-Sketch.
* **Fine-Grained:** DTD, Flower102, Caltech101, Aircraft, Pets, UCF101, Cars, EuroSAT, SUN397, Food101.

## 🚀 Usage

### 1. Optimization-Free Inference (SE)

To run the Self-Ensembling (SE) strategy as a lightweight, training-free TTA method:

```bash
python instance_tta.py --test_sets DTD --algorithm se -a ViT-B/16
```

### 2. Unified Self-Ensembling (USE)

To run the full optimization-based USE framework:

```bash
python instance_tta.py --test_sets DTD --algorithm use -a ViT-B/16
```

## 📜 Citation

If you find our work helpful for your research, please consider citing our paper:

```bibtex
@inproceedings{jiang2026use,
  title={USE: A Unified Self-Ensembling Framework for Test-Time Prompt Tuning},
  author={Jiang, Siru and Liang, Jian and He, Ran and Tan, Tieniu},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## 🙏 Acknowledgments

This repository is built upon the codebases of [VLM-TTA](https://github.com/TomSheng21/tta-vlm?tab=readme-ov-file). We thank the authors for releasing their excellent work. 
