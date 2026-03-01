# SHIFT Paper Memory

## Paper Title
SHIFT: Steering High-level Image Features with Text

## Venue
ECCV 2026

## Core Thesis
Pretrained ViTs exhibit saliency bias (collapse to the "loudest" concept). SHIFT injects text conditioning into a frozen ViT via sparse, tanh-gated cross-attention layers to steer visual features without degrading them. This achieves a Pareto improvement: high text steerability + preserved feature quality.

## Research Question
Can text conditioning steer the semantics of visual representations without degrading their quality?

## Three Desirable Properties
1. **Global steerability**: text steers the global feature vector (measured by kNN retrieval on synthetic edited scenes)
2. **Dense attention routing**: self-attention routes to text-relevant tokens (measured by mosaic benchmark / PASCAL-VOC AUC)
3. **Feature quality preservation**: features stay in the base ViT's embedding space (measured by classification probes, Pareto frontier)

## Architecture Details
- **Visual encoder**: Frozen DINOv2 ViT-B/14 (all original params frozen)
- **Text encoder**: Frozen RoBERTa-Large, L2-normalized outputs
- **Multimodal adapter**: 2-layer MLP projecting text embeddings to visual dim
- **Gated cross-attention**: Flamingo-style tanh-gated dense cross-attention (NOT sigmoid)
  - Inserted every 3rd block from block 2 (blocks 2, 5, 8, 11 for 12-block ViT-B)
  - Final block always remains unmodified frozen ViT
  - Zero-init: tanh(0)=0 so model = frozen ViT at initialization
- **Training head**: Linear head for binary patch-level segmentation (discarded at inference)

## Training
- **Objective**: Binary referential segmentation at ViT patch level (soft cross-entropy)
- **Loss**: Only segmentation loss (no distillation)
- **Datasets**: RefCOCO, RefCOCO+, RefCOCOg, Visual Genome (MDeTr preprocessed), Mapillary Vistas

## Key Results
- **kNN retrieval on edited scenes**: 96.6% indoor / 91.3% outdoor (vs DINOv2 near random chance)
- **Mosaic AUC** (PASCAL-VOC): 50.4 vs DINOv2's 16.7 (33.7% improvement)
- **Personalized reps (PODS)**: With detailed descriptions, surpasses fine-tuned DINOv2 (54.4 vs 48.0 PR-AUC)
- **Anomaly detection (MVTec)**: 64.2 AuPRO (vs CLIPSeg 34.0, SAM2 48.2)
- **Pareto frontier**: Gate scaling factor 0.6 is optimal operating point

## Key Baselines
- **Unsteerable encoders**: DINOv2, SigLIP, MAE
- **Steerable but degraded**: SAM2, GroundingDINO
- **Late fusion**: SigLIP with add/concat (doesn't help)
- **MLLMs**: InternVL-1B, Qwen3-2B (features not competitive)

## Paper Structure
1. Introduction (drafted)
2. Related Work (empty)
3. Methodology (drafted: Architecture + Training Objective + Training Data)
4. Experiments (partially drafted: baselines, kNN, mosaics, PODS, t-SNE, Pareto)
5. Ablations (empty)
6. Conclusion (empty)

## Writing Style (from SCIENTIFIC_WRITING_SKILL.MD)
- **BLUF**: lead every paragraph/section with its main claim
- **One sentence per line** in LaTeX source
- **Characters as subjects**, actions as verbs (no nominalizations)
- **Old before new** information flow
- **No em dashes**: use parentheses, colons, semicolons instead
- **Concise**: cut filler, no vague quantifiers
- **Figure captions**: "so what" not "what"

## Contributions (as listed in intro)
1. SHIFT method (tanh-gated cross-attention in frozen ViTs, trained on referential segmentation)
2. Pareto improvement (steerability + quality, bridging the gap)
3. Extensive evaluation (conditional retrieval, dense prediction, PODS; across DINOv2, SigLIP, MAE)
4. Applications (zero-shot anomaly detection at 64.2 AuPRO on MVTec; personalized object discrimination)

## Important Notes
- The gating is **tanh** (not sigmoid) -- experiments section was corrected for consistency
- Segmentation head is a **proxy training signal only** -- discarded at inference
- The output of interest is the **steered visual representation**, not segmentation masks
- "SAM3" in internal notes = SAM2 in the paper citations
