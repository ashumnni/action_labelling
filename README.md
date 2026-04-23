# Action Labelling Engine - Session Results

This repository contains the outputs and tools used during the automated action labeling session on **April 23, 2026**. The focus was on processing egocentric video data for dual-mode action recognition.

## 🛠 Actions Completed

### 1. Environment Optimization
- Installed and configured `openai`, `python-dotenv`, `tqdm`, and `opencv-python`.
- Updated `requirements.txt` to streamline future setup.

### 2. Dual-Mode Temporal Action Labeling (TAL)
We processed **`SortingVegetables-1.mov`** using GPT-4o Vision to generate two distinct types of labels:
- **Descriptive Pass**: 10-12 word high-fidelity human activity descriptions.
- **Robot TAL Pass**: 6-8 word imperative command contexts for robotic control.
- **Master Ledger**: Integrated results into the centralized `master_results.json`.

### 3. Visual Verification
Generated a high-resolution visualization with a dynamic HUD overlay:
- **File**: `SortingVegetables-1_labeled.mp4`
- **Features**: Real-time action labels, confidence metrics, and temporal tracking.

## 🚀 How to Run (Session Scripts)

### Labeling
To reproduce the labeling results:
```bash
# For Robot Commands (TAL)
python gpt_labeler.py --video SortingVegetables-1.mov --format tal --output sorting_tal_dense_v1.json

# For Descriptive Labels
python gpt_labeler.py --video SortingVegetables-1.mov --format descriptive --output sorting_descriptive_v1.json
```

### Visualization
To generate the labeled video:
```bash
python visualizer.py --video SortingVegetables-1.mov --master sorting_tal_dense_v1.json --output SortingVegetables-1_labeled.mp4
```

## 📂 Session Outputs
- [`sorting_descriptive_v1.json`](sorting_descriptive_v1.json)
- [`sorting_tal_dense_v1.json`](sorting_tal_dense_v1.json)
- `SortingVegetables-1_labeled.mp4` (Visualization)
- `master_results.json` (Consolidated Ledger)
