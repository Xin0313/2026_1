# SCHPose

**SCHPose: Symmetry-aware Continuous-discrete Hybrid encoding for single-stage 6D Pose Estimation**

A complete RGB-only, single-stage 6D object pose estimation framework targeting YCB-Video, Linemod-Occluded, and T-LESS datasets.

## Project Structure

```
SCHPose/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── default.yaml
│   ├── ycbv.yaml
│   ├── lmo.yaml
│   └── tless.yaml
├── datasets/
│   ├── __init__.py
│   ├── bop_dataset.py
│   ├── augmentation.py
│   └── surface_codebook.py
├── models/
│   ├── __init__.py
│   ├── backbone.py
│   ├── heads.py
│   ├── hybrid_decoder.py
│   ├── diff_pnp.py
│   └── schpose.py
├── losses/
│   ├── __init__.py
│   ├── seg_loss.py
│   ├── code_loss.py
│   ├── offset_loss.py
│   ├── uncertainty_loss.py
│   ├── pose_loss.py
│   └── total_loss.py
├── utils/
│   ├── __init__.py
│   ├── geometry.py
│   ├── symmetry.py
│   ├── metrics.py
│   └── visualization.py
└── tools/
    ├── train.py
    ├── test.py
    ├── eval_bop.py
    ├── generate_codebook.py
    └── demo.py
```

## Installation

```bash
cd SCHPose
pip install -e .
# or
pip install -r requirements.txt
```

## Data Preparation

Download datasets in BOP format from https://bop.felk.cvut.cz/datasets/

```
data/
├── ycbv/
│   ├── models/          # obj_000001.ply ... obj_000021.ply
│   ├── train_pbr/
│   ├── train_real/
│   └── test/
├── lmo/
│   ├── models/
│   └── test/
└── tless/
    ├── models/
    └── test_primesense/
```

## Generate Surface Codebooks

```bash
python tools/generate_codebook.py --config configs/ycbv.yaml
python tools/generate_codebook.py --config configs/lmo.yaml
python tools/generate_codebook.py --config configs/tless.yaml
```

## Training

**Single GPU:**
```bash
python tools/train.py --config configs/ycbv.yaml
```

**Multi-GPU (DDP):**
```bash
torchrun --nproc_per_node=4 tools/train.py --config configs/ycbv.yaml
```

Training is split into two phases:
- **Phase 1** (epochs 1–80): Train segmentation, discrete code, continuous offset, and uncertainty heads
- **Phase 2** (epochs 81–200): Unfreeze differentiable PnP, add pose loss for end-to-end fine-tuning

## Testing

```bash
python tools/test.py --config configs/ycbv.yaml \
                     --checkpoint outputs/ycbv/best_checkpoint.pth
```

## Demo

```bash
python tools/demo.py --config configs/ycbv.yaml \
                     --checkpoint outputs/ycbv/best_checkpoint.pth \
                     --image path/to/image.jpg \
                     --K "fx,fy,cx,cy" \
                     --obj_id 1
```

## BOP Evaluation

```bash
python tools/eval_bop.py --results_path outputs/ycbv/bop_results.csv \
                          --dataset ycbv
```

## Architecture

```
RGB Image [B,3,H,W]
     │
     ▼
ConvNeXt-Tiny + FPN
     │  {P2(1/4), P3(1/8), P4(1/16), P5(1/32)}
     │
     ▼ P2 [B,256,H/4,W/4]
┌────┴────────────────────────────────┐
│         5 Decoupled Heads           │
│  (a) Seg/Det Head                   │
│  (b) Discrete Code Head (coarse+fine│
│  (c) Continuous Offset Head (u,v)   │
│  (d) Uncertainty Head (log σ²)      │
│  (e) Symmetry Class Head            │
└────┬────────────────────────────────┘
     │
     ▼
Hybrid Decoder
  (BlockID + continuous offset) → 3D surface points
     │
     ▼
Differentiable Weighted PnP
  - Importance sampling (top-K by confidence)
  - Weighted EPnP initialization
  - Gauss-Newton refinement (5 steps)
  - Symmetry enumeration → best pose
     │
     ▼
  (R, t) per object
```

## Loss Function

```
L_total = 2.0·L_seg + 1.0·L_code + 5.0·L_uv + 1.0·L_unc + 10.0·L_pose + 0.5·L_consist
```

| Component | Description |
|-----------|-------------|
| L_seg | Focal + Dice segmentation loss |
| L_code | Hierarchical cross-entropy (coarse + fine) with label smoothing |
| L_uv | Charbonnier loss for within-block (u,v) coordinates |
| L_unc | NLL uncertainty: ‖e‖²/σ² + log σ² |
| L_pose | Symmetry-aware ADD(-S) pose loss (Phase 2) |
| L_consist | Hierarchical consistency regularization |

## Expected Results

| Dataset | Method | ADD(-S) AUC |
|---------|--------|-------------|
| YCB-V | SCHPose | ~75% |
| LM-O | SCHPose | ~55% |
| T-LESS | SCHPose | ~50% |

*Results are approximate targets; actual results depend on training data and hyperparameters.*
