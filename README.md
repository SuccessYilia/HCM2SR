# HCM$^2$SR: Hybrid Contrastive Multi-Scenario Recommendation

## Overview
HCM$^2$SR is a state-of-the-art recommendation model designed to capture both shared and specific information across multiple scenarios using contrastive learning. This repository contains the full implementation of the model, experimental setups, and instructions for reproducing the results presented in our paper.

## Features
- **Hybrid Contrastive Learning**: Efficiently captures shared and unique characteristics across different scenarios.
- **Adaptive Knowledge Transfer**: Facilitates the transfer of knowledge across various stages to tackle sparsity issues in long-path conversions.
- **Self-Supervised Signals**: Alleviates data sparsity by utilizing self-supervised learning techniques.

## Installation
To get started, clone this repository and install the required packages:

```bash
git clone https://github.com/SuccessYilia/HCM2SR.git
cd HCM2SR
pip install -r requirements.txt

python train.py --config config.yaml

python evaluate.py --model_path <path_to_model>

