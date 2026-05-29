# Electrolyte Inspection System

Industrial vision system for detecting electrolyte uniformity in lithium coin cell batteries. Uses a tilted light path + adaptive exposure fusion optical setup with deep learning (ResNet34/EfficientNet/ConvNeXt) for binary classification (OK/NG).

## Features

- **Dual-camera detection**: Miscibility (互溶) + Precipitation (分层) parallel inference
- **Multi-model ensemble**: Soft voting across ResNet34, EfficientNet-B0, ConvNeXt-Tiny
- **A/B testing framework**: Gradual rollout, automatic promotion based on accuracy
- **Active learning loop**: Uncertainty-based sample collection, auto-retraining
- **Temperature-scaled calibration**: Smoothed probabilities, data-driven threshold
- **TTA (Test-Time Augmentation)**: 4-view voting for robust inference
- **PLC integration**: TCP communication with production line PLC
- **Cold-tone filter**: Training-inference alignment for camera color profile

## Project Structure

```
electrolyte-inspection/
├── src/
│   ├── detection/          # Models, inference, transforms
│   ├── training/           # Trainer, dataset, calibration
│   ├── active_learning/    # Uncertainty queue, auto-retrain scheduler
│   ├── ensemble/           # Multi-model voting, A/B testing
│   ├── api/                # FastAPI prediction server
│   ├── plc/                # PLC TCP server, camera acquisition
│   └── utils/              # Configuration
├── scripts/                # train_all.py, evaluate.py
├── tests/
├── configs/
└── model_registry/         # Saved models + calibration configs
```

## Quick Start

```bash
pip install -r requirements.txt

# Train all models
python scripts/train_all.py --raw_dir /path/to/training/data

# Evaluate a model
python scripts/evaluate.py --model_path model_registry/resnet34.pth

# Run inference on single image
python -c "
from PIL import Image
from src.detection.inference import Predictor
p = Predictor.from_checkpoint('model_registry/resnet34.pth')
print(p.predict(Image.open('test.jpg')))
"
```

## Training Data Format

```
raw_data_hierarchical/
├── hierarchical/       # NG samples (electrolyte shows layering)
└── non_hierarchical/   # OK samples (uniform electrolyte)
```

## Model Performance

| Model | Accuracy | AUC | Params |
|-------|----------|-----|--------|
| ResNet34 | 94.1% | 1.000 | 21M |
| EfficientNet-B0 | - | - | 5.3M |
| ConvNeXt-Tiny | - | - | 29M |

*Note: Results on 162-sample dataset. Production performance depends on data quality and quantity.*
