# HCM$^2$SR: Hybrid Contrastive Multi-Scenario Recommendation

## Overview
HCM$^2$SR is a state-of-the-art recommendation model in multi-scenario and multi-task learning domains, which is designed to capture both shared and specific information across multiple scenarios using contrastive learning. Furthermore, the adaptive knowledge transfer network facilitates the transfer of knowledge across various sequential-dependence tasks to tackle sparsity issues in long-path conversions. This repository contains the implementation of the model, experimental setups, and instructions. We are committed to keeping this repository up-to-date with the latest implementations and improvements. We will continue to update and improve this repository with new features, optimizations, and bug fixes based on ongoing research and feedback from the community.

<img width="563" alt="截屏2024-11-04 08 49 15" src="https://github.com/user-attachments/assets/6ad279bc-8f82-46a6-9502-4aa0882ace27">


## Modules
- **Hybrid Contrastive Learning**: Efficiently captures shared and unique characteristics across different scenarios. The self-supervised signals alleviate data sparsity by utilizing self-supervised learning techniques. Considering that different scenarios contribute differently to each other’s representation capability, a scenario-aware multi-gate network is further designed to explicitly evaluate the significance of knowledge from other scenarios to the representations in the current scenario.
- **Adaptive Knowledge Transfer**: Facilitates the transfer of knowledge across various stages to tackle sparsity issues in long-path conversions.


## Installation
To get started, clone this repository and install the required packages:

```bash
git clone https://github.com/SuccessYilia/HCM2SR.git
cd HCM2SR
pip install -r requirements.txt

python train.py --config config.yaml

python evaluate.py --model_path <path_to_model>

