# Vision Transformer Attention Analysis: Structural Effects of Fine-Tuning

## Overview

This repository contains the research code for a Master's thesis investigating how fine-tuning alters attention structure in Vision Transformers. The work focuses on quantifying spatial properties of attention patterns, not on optimizing classification performance or claiming semantic understanding from attention weights.

The study compares attention patterns between a baseline model (pretrained backbone with trained classification head) and a fully fine-tuned model on MIT Indoor-67 scene classification. Analysis is conducted post-hoc after training convergence to characterize steady-state attention structure.

**Core principle**: Attention weights reveal computational dependencies, not complete explanations. This project measures structural changes in attention distributions without claiming those changes reflect improved reasoning.

## Research Question

**Does fine-tuning on scene classification systematically alter the spatial structure of attention dependencies in Vision Transformers, and can these changes be quantified using spatial statistical methods?**

This question emphasizes measurable structure over semantic interpretation. We quantify whether and how attention distributions change spatially, not whether those changes are "better" in a semantic sense.

## Methodology

### Experimental Design

**Model**: ViT-B/16 (timm), pretrained on ImageNet-21k  
**Dataset**: MIT Indoor-67 (15,620 images, 67 classes, stratified 70/15/15 split)  
**Conditions**:
- Baseline: Frozen pretrained backbone, trained classification head
- Fine-tuned: All parameters trainable, layer-wise learning rates

Both conditions trained with identical augmentation (horizontal flip, small rotations, color jitter—no CutMix/Mixup), label smoothing, and cosine schedules. Training details in `config.py`.

### Attention Extraction

Attention weights extracted from all 12 transformer blocks after softmax, before dropout. Implementation uses factory functions to prevent closure bugs and explicit per-layer indexing. The classification token ([CLS]) is excluded from spatial metrics as it lacks spatial meaning.

### Analysis Metrics

Three spatial statistics characterize attention structure layer-wise:

**1. Dependency Distance**  
Average Euclidean distance between attending patch pairs, weighted by attention probability, normalized by image diagonal. Measures spatial extent of attention (local vs. long-range).

**2. Spatial Coherence (Moran's I)**  
Spatial autocorrelation measuring whether high-attention patches cluster (positive I) or scatter (near zero). Uses 4-connectivity for spatial neighbors.

**3. Attention Entropy**  
Shannon entropy of attention distributions. Measures concentration (low entropy = focused) vs. diffusion (high entropy = distributed).

Statistical significance assessed via paired t-tests across layers. Grad-CAM generated for qualitative inspection only.

### What These Metrics Do NOT Measure

These metrics characterize spatial distributions but **do not**:
- Prove semantic understanding (localized ≠ object-focused; coherent ≠ meaningful)
- Establish causality (correlation with fine-tuning ≠ causal importance)
- Validate attention correctness (no ground truth for "correct" attention)
- Explain decisions (attention is one component; feedforward layers, residuals, value transformations also critical)

Grad-CAM shows gradient correlations, not causal features. It is illustrative, not evidential.

## Repository Structure

```
├── config.py                      # Hyperparameters
├── data/
│   ├── dataset.py                # MIT Indoor-67 loader
│   └── splits.py                 # Stratified splits
├── models/
│   └── vit.py                    # ViT with attention extraction
├── analysis/
│   └── attention.py              # Spatial metrics implementation
├── experiments/
│   ├── train_baseline.py         # Train baseline
│   ├── train_finetune.py         # Train fine-tuned
│   └── compare.py                # Compute metrics
└── outputs/
    ├── checkpoints/              # Model weights (.pth)
    ├── metrics/
    │   └── attention_comparison.json  # Layer-wise results
    ├── visualizations/
    │   └── gradcam_samples/      # Grad-CAM examples
    └── summary.md                # Detailed interpretation
```

## Understanding the Results

### Output Files

**`outputs/metrics/attention_comparison.json`**  
Layer-wise quantitative results for all three metrics (12 values per metric per condition). This is the primary data file.

**`outputs/summary.md`**  
Structured interpretation document containing:
- Observed patterns in metrics (if any systematic differences exist)
- Statistical significance tests
- Cross-metric relationships
- Explicit interpretation limitations
- Guidance on reading results within constraints

**Read `summary.md` before interpreting numerical results.** It provides essential context on what observed differences do and do not mean.

**`outputs/visualizations/gradcam_samples/`**  
Qualitative Grad-CAM examples. These illustrate potential differences but are not quantitative evidence.

### Interpretation Guidance

When examining results:
1. Check for layer-wise trends (shallow vs. deep layers)
2. Assess statistical significance (reported in `summary.md`)
3. Look for cross-metric consistency (do all three metrics change coherently?)
4. Remember: all findings are correlational, not causal
5. Grad-CAM is for inspection, not validation

**All interpretation must respect the limitations documented in `summary.md`.**

## Reproducibility

### Setup
```bash
pip install torch torchvision timm scipy numpy matplotlib tqdm
```

Download MIT Indoor-67 to `data/MIT_Indoor_67/Images/`

### Training
```bash
python experiments/train_baseline.py
python experiments/train_finetune.py
```

### Analysis
```bash
python experiments/compare.py \
    --baseline outputs/checkpoints/baseline_best.pth \
    --finetuned outputs/checkpoints/finetuned_best.pth
```

**Reproducibility notes**: Attention metrics are deterministic given fixed weights. Classification accuracy may vary slightly across hardware due to non-deterministic CUDA operations. Training uses seed=42 and deterministic settings for analysis reproducibility.

**Computational requirements**: Training ~6 hours per model on A100 (40GB), extraction ~45 minutes per model, metrics ~30 minutes CPU.

## What This Project Does NOT Claim

This work explicitly **does not** claim:

1. **Attention equals explanation** – Attention weights are one component among many (value transformations, feedforward layers, residuals)
2. **Attention correctness is validated** – No ground truth for "correct" attention exists
3. **Fine-tuning improves attention quality** – We observe structural changes, not quality improvements
4. **Grad-CAM identifies causal features** – Grad-CAM shows correlations, not causes
5. **Results generalize beyond scope** – Findings specific to ViT-B/16 on MIT Indoor-67
6. **Causality is established** – All findings are correlational; attention changes may be incidental to performance
7. **Attention was supervised** – Analysis is post-hoc; no attention-based training objectives used

## Limitations

**Scope**: Single architecture (ViT-B/16), single dataset (MIT Indoor-67), single comparison (baseline vs. fine-tuned)

**Analysis depth**: Only attention weights analyzed; complete understanding requires examining all model components

**Statistical power**: Analysis on ~1,600 test samples; rare patterns may be missed

**Causation**: Observed differences correlate with fine-tuning but causal relationships unestablished

**Generalization**: Results may not transfer to other architectures, datasets, or tasks

See `outputs/summary.md` for detailed limitations and interpretation constraints.

## Intended Use

This codebase is designed for:
- Thesis examination and peer review
- Reproducibility and methods validation
- Reference implementation of spatial attention metrics
- Educational purposes in interpretability research

**Not designed for**: Production deployment, transfer to new datasets without modification, achieving state-of-the-art accuracy, or real-time analysis.

## Citation

```
[Thesis citation to be added upon publication]
```

## License

MIT License. Code provided for research and educational purposes.

---

**Note to reviewers**: This repository prioritizes scientific rigor. All claims include explicit caveats. Consult `outputs/summary.md` for detailed interpretation guidance and statistical analysis.