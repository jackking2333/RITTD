# Repository Name

This repository contains the source code and comparison methods for the RITTD (Reasoning-Induced Toxic Text Detection for Large Language Models) method.

## Contents

The repository consists of the following:

- `src/`: This directory contains three files corresponding to the different steps of the RITTD method:
  - `generate_reason.py`: Code for generating analysis using generative models in the RITTD method.
  - `inference.py`: Code for performing predictions using the RITTD method.
  - `trainwithlora.py`: Code for training with difficult samples using the LORA  method.

- `comparison_methods/`: This folder includes four different comparison methods.
  - `discriminative_model.py`: Implementation of the discriminative model method for comparison.
  - `perspective.py`: Integration with the Perspective API for comparison.
  - `ICL.py`: Implementation of the ICL (In-Context Learning) method for comparison.
  - `pretrained_classifier.py`: This directory contains the pretrained detection model used in the discriminative model.

## Usage

To use the RITTD method and comparison methods, follow the instructions below:

1. Navigate to the `src/` directory.
2. Run `generate_reason.py` to generate analysis using the generative models.
3. Use `inference.py` to perform predictions using the RITTD method.
4. Execute `train_with_lora.py` to train with difficult samples using the LORA algorithm.

For the comparison methods, refer to the respective folders and follow the provided instructions.


