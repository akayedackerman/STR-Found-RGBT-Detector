# STR-Found: Spatio-Temporal RGBT Tiny Object Detector



Custom multi-modal implementation components for the STR-Found object detection framework.



## 📁 Core Code Architecture

- `models/`: DualResNetSTRFound framework featuring STIT transformer tracking blocks and CDAF multi-modal layers.

- `loss/`: Fortified batch-shuffled InfoNCE Temporal Contrastive Loss block.

- `pipeline/`: Custom temporal multi-frame RGBT data pair loaders.

- `auxiliary_codes/`: Analytical labels and target inspection toolkits.

- `rgbt_str_found_cascade.py`: Main execution training configuration blueprint.



## 🚀 Execution Guide

To run these layers within an MMDetection repository installation, place the contents of this folder into your workspace root and run:

```bash

PYTHONPATH="$(pwd)" python tools/train.py rgbt_str_found_cascade.py
