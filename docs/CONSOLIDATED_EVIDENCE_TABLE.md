# Consolidated Evidence Table

Validation and test metrics are in separately labelled columns and are never
combined. Single training seed (0) per arm — no multi-seed claim is supported.
Historical Tesla-T4 measurements are not represented here; they live in their own
labelled table and are never differenced against these A100-trained results.

| Logical ID | Physical | Model | Total params | Trainable | Status | Best Val Top-1 | Clean Test Top-1 | Easy Test Top-1 | Moderate Test Top-1 | Hard Test Top-1 | Clean→Hard drop | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | A0 | YOLOv8n-cls RGB baseline | 1,488,247 | 1,488,247 | COMPLETED | 0.9976036 | — | — | — | — | — |  |
| A1 | A1 | PE-only | 1,488,247 | 1,488,247 | COMPLETED | 0.9968847 | — | — | — | — | — |  |
| A2 | A2 | TF-only (ViT-B/16) | 87,289,243 | 1,490,587 | INCOMPLETE (26/50) | — | — | — | — | — | — | stalled at epoch 26/50; excluded from evaluation |
| A3 | A3 | PE+TF, no AE (= F3) | 87,289,243 | 1,490,587 | COMPLETED | 0.9966451 | — | — | — | — | — |  |
| A4 | A4 | RGB + image-space AE | 1,747,290 | 1,747,290 | COMPLETED | 0.9962856 | — | — | — | — | — |  |
| A5 | A5 | Original AE-TFPE full (= F5) | 87,549,123 | 1,750,467 | COMPLETED | 0.9950875 | — | — | — | — | — |  |
| B1 | B1 | ResNet-50 baseline | 23,587,943 | 23,587,943 | NOT YET RUN | — | — | — | — | — | — | skipped by MAX_TRAIN_PARAMS=20,000,000; not scientifically impossible |
| B2 | B2 | EfficientNet-B0 baseline | 4,057,507 | 4,057,507 | COMPLETED | 0.9988018 | — | — | — | — | — |  |
| B3 | B3 | ViT-B/16 baseline | 85,828,647 | 85,828,647 | NOT YET RUN | — | — | — | — | — | — | skipped by MAX_TRAIN_PARAMS=20,000,000; the backbone the Original method uses |
| D1 | D1 | Original AE fusion, clean objective | 87,549,123 | 1,750,467 | COMPLETED | 0.9944884 | — | — | — | — | — |  |
| E3 | E3 | Efficient PE+TF, no AE | 1,637,947 | 1,488,427 | COMPLETED | 0.9964055 | — | — | — | — | — | confounded AE control: also changes the fusion space (grid -> image) |
| E5 | E5 | Efficient AE-TFPE (C2-28) | 1,716,586 | 1,567,066 | COMPLETED | 0.9877786 | — | — | — | — | — |  |
| E7 | E7 | Efficient AE-TFPE (C2-7) | 2,544,634 | 1,593,610 | COMPLETED | 0.8436377 | — | — | — | — | — |  |
| F1 | F1 | Addition fusion | 87,289,216 | 1,490,560 | COMPLETED | 0.9966451 | — | — | — | — | — |  |
| F2 | F2 | Concatenation fusion | 87,289,648 | 1,490,992 | COMPLETED | 0.9971244 | — | — | — | — | — | the only arm that modifies the classifier stem |
| F3 | A3 | reuses A3 | — | — | LOGICAL_REUSE | — | — | — | — | — | — |  |
| F4 | F4 | Attention fusion | 87,289,288 | 1,490,632 | COMPLETED | 0.9955667 | — | — | — | — | — |  |
| F5 | A5 | reuses A5 | — | — | LOGICAL_REUSE | — | — | — | — | — | — |  |
| F5_clean | D1 | reuses D1 | — | — | LOGICAL_REUSE | — | — | — | — | — | — |  |
| M1 | M1 | Legacy LUT control | 1,488,247 | 1,488,247 | COMPLETED | 0.9958064 | — | — | — | — | — |  |
| M2 | M2 | Photometric (gamma) control | 1,488,247 | 1,488,247 | COMPLETED | 0.9976036 | — | — | — | — | — |  |
| M3 | M3 | Augmentation control | 1,488,247 | 1,488,247 | COMPLETED | 0.9968847 | — | — | — | — | — |  |

Source: `results/evaluation` prediction records; training facts from `docs/RUN_INVENTORY.md`.
