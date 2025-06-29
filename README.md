# Patch-Based Spoof Detection

Patch-Based Spoof is a deep learning framework for face anti-spoofing using patch-based convolutional neural networks. This repository provides modular code for training, evaluating, and experimenting with patch-based and keypoint-based models for presentation attack detection (PAD).

## Features
- Patch-based CNN models for spoof detection
- DeepPix and its variants (HSV, six-channel, extended)
- Keypoint-based experiments
- Modular dataloaders and trainers
- Evaluation scripts and metrics (APCER, BPCER, ACER)
- Jupyter notebooks for exploration and visualization

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mdhasanai/patch-based-spoof.git
   cd patch-based-spoof
   ```
2. (Recommended) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Create a `requirements.txt` if not present, listing packages like torch, torchvision, numpy, pandas, etc.)*

## Usage
### Training a Patch-Based Model
```bash
python scripts/patch_train.py --config <config-file>
```

### Training DeepPix Model
```bash
python scripts/deeppix_train.py --config <config-file>
```

### Evaluation
```bash
python scripts/eval_keypoint.py --model <model-path> --data <data-path>
```

### Visualization
Explore the provided Jupyter notebooks in `scripts/` for filter analysis, heatmap visualization, and more.

## Project Structure
```
patch-based-spoof/
├── dataloaders/         # Data loading utilities
├── models/              # Model architectures (DeepPix, Patch-based CNN)
├── scripts/             # Training, evaluation, and analysis scripts
│   ├── metrics/         # Metric calculation scripts
│   ├── eval/            # Evaluation utilities
├── trainers/            # Training logic for different models
├── utils/               # Helper functions (evaluator, etc.)
├── README.md            # Project documentation
```

## Training & Evaluation
- Configure your training and evaluation parameters in the respective scripts or config files.
- Use the provided scripts for patch-based or keypoint-based experiments.
- Metrics such as APCER, BPCER, and ACER are available in `scripts/metrics/`.

## Contributing
Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- Inspired by DeepPix and related face anti-spoofing research.
- Thanks to all contributors and the open-source community.

---
For questions or support, please open an issue on GitHub.
