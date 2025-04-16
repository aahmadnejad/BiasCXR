# BiasCXR: Debiasing Chest X-Ray Datasets with Up-sampling and Down-sampling Techniques

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow 2.8+](https://img.shields.io/badge/tensorflow-2.8+-orange.svg)](https://www.tensorflow.org/)

## Overview

BiasCXR addresses algorithmic bias in deep learning models for chest X-ray (CXR) classification, which can exacerbate healthcare disparities across demographic groups. This project implements strategic up-sampling and down-sampling techniques to mitigate biases in the MIMIC-CXR dataset across sensitive attributes including race, gender, age, and insurance status.

Our approach improves fairness metrics while maintaining diagnostic accuracy, contributing to the development of more equitable AI systems in medical imaging.

## Dataset

This project uses the [MIMIC-CXR Dataset (v2.1.0)](https://physionet.org/content/mimic-cxr/2.1.0/), one of the largest publicly available collections of chest radiographs with corresponding reports from the Beth Israel Deaconess Medical Center. The dataset contains:

- Approximately 370,000 chest X-rays 
- Rich demographic information (race, gender, age, insurance status)
- Labels for 14 common radiographic findings

Access requires completion of the CITI training to ensure appropriate use of sensitive healthcare data.

## Key Features

- **Comprehensive Bias Analysis**: Detailed evaluation of distribution imbalances across demographic attributes and their correlation with pathological findings
- **Advanced Rebalancing Techniques**:
  - Cluster-based downsampling for overrepresented categories
  - Controlled data augmentation for underrepresented groups
- **Fairness Evaluation**: Multiple metrics to assess model performance across demographic groups
- **Trade-off Analysis**: Examination of the balance between overall performance and algorithmic fairness

## Repository Structure

```
BiasCXR/
├── main.py             # Main script to run training and evaluation
├── debiasing.py        # Implementation of debiasing techniques
├── model.py            # Neural network architecture and training
├── plotting.py         # plotting functions for the entire project
├── processor.py        # processing functions such as creating debiased dataset
├── inspectTFrecord.py  # read and see whats in tfrecords
├── cuda_check.py       # check if CUDA is available
├── cxr.py              # The CXR pytorch dataset class
├── store.py            # Configuration settings using dataclass
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── LICENSE             # MIT License
```

** all outputs will save in a folder called output along with the tensorboard logs **

## Installation

```bash
git clone https://github.com/aahmadnejad/BiasCXR.git
cd BiasCXR

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Configuration

The project uses a centralized configuration system in `store.py` using Python's `@dataclass`. This contains all the model parameters and settings including:

- Model architecture dimensions (embedding size, hidden layers dimensions)
- Training parameters (batch size, learning rate, epochs)
- File paths for saving models
- Labels for the 14 chest X-ray conditions
- Age binning configuration for demographic analysis

To modify the configuration, edit the `Configs` class in `store.py` rather than changing parameters throughout the codebase. This ensures consistency across all components of the project.

Key configurations include:
- Embedding dimension: 1376
- Hidden layer sizes: 512 and 256 neurons
- Batch size: 16
- Learning rate: 0.001
- Age groups: 0-20, 20-40, 40-60, 60-80, 80+

## Model Architecture

The deep learning model implemented in `model.py` has the following architecture:

```
Neural Network Architecture:

Input Layer (1376 features) 
      ↓
Dense Layer (512 neurons, ReLU activation)
      ↓
Dropout (rate=0.3)
      ↓
Dense Layer (256 neurons, ReLU activation)
      ↓
Dropout (rate=0.3)
      ↓
Output Layer (14 neurons, sigmoid activation)
```

This architecture is designed for multi-label classification of 14 common radiographic findings from chest X-ray embeddings.


### Running from Command Line

The simplest way to use this repository is through the command-line interface:

```bash
# Basic usage
python main.py --csv_path /path/to/mimic_metadata.csv
```


## Dependencies

```
tensorflow>=2.8.0
pytorch>=2.6.0
numpy>=1.19.5
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.2
tensorboard>=2.19.0
tqdm==4.67.1
```

Additionally, please cite the MIMIC-CXR dataset:

```bibtex
@article{johnson2019mimic,
  title={MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports},
  author={Johnson, Alistair EW and Pollard, Tom J and Berkowitz, Seth J and Greenbaum, Nathaniel R and Lungren, Matthew P and Deng, Chih-ying and Mark, Roger G and Horng, Steven},
  journal={Scientific data},
  volume={6},
  number={1},
  pages={317},
  year={2019},
  publisher={Nature Publishing Group}
}
```

## Results

[auc_comparison](GitAssets/auc_comparison.png)
[fnr_comparison](GitAssets/fnr_comparison.png)
[overall_fnr_improvement](GitAssets/overall_fnr_improvement.png)
[overall_fpr_improvement.png](GitAssets/overall_fpr_improvement.png)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The MIMIC-CXR dataset is made available by the Beth Israel Deaconess Medical Center
- Access provided through PhysioNet
- The code report is [FairnessNBiass](FairnessNBiass.pdf) file.
