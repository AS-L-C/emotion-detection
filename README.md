# Emotion Detection

This repository explores the use of EEG and time-series foundation models for emotion classification on an EEG emotion dataset.

The project supports three foundation models:

- **REVE** — [Paper](https://arxiv.org/pdf/2510.21585)
- **CBraMod** — [Paper](https://arxiv.org/pdf/2412.07236)
- **UniShape** — [Paper](https://arxiv.org/pdf/2601.06429)

## Overview

The goal of this repository is to classify emotions from EEG data by leveraging recent foundation models as either feature extractors or fine-tuned classifiers.

### Model Selection

**REVE** and **CBraMod** were selected because they are reported as the top-performing EEG foundation models in the REVE paper.

**UniShape** was selected because it has been shown to outperform other time-series foundation models when used as a feature extractor for downstream classification tasks.

When multiple model sizes were available, the smallest model variant was used to reduce computational cost.

## Modeling Strategy

The models are used as follows:

| Model | Usage |
|---|---|
| **REVE** | Feature extractor with a trained classification head |
| **CBraMod** | Feature extractor with a trained classification head |
| **UniShape** | Fine-tuned directly on the emotion dataset |

For REVE and CBraMod, the pretrained models are kept frozen and used to extract representations. A classification head is then trained on top of the extracted features.

For UniShape, the model is fine-tuned end-to-end on the emotion dataset (but keeping the shape extractor fixed).

## Main Entry Point

The main workflow is implemented in:

```text
code/main_notebook.ipynb
```

The notebook performs the following steps:

1. Sets up the environment
2. Preprocesses the dataset using model-specific pipelines based on the corresponding papers
3. Optionally runs hyperparameter optimization using Optuna
4. Trains the model using the best hyperparameter configuration
5. Computes predictions on the test set

## Prediction Results

Test predictions are provided for the `CBraMod` model trained for 50 epochs on the full training dataset after hyperparameter optimization.

The prediction file is available at:

```text
emotion-detection/results/test_predictions/cbramod/best_model_predictions.csv
```
This repository supports all three foundation models described above. Each model can be run and trained on small datasets, for example, by setting DEBUG = True.

However, due to hardware limitations, only CBraMod was trained on the full dataset. The largest available compute resource was a single NVIDIA H100 GPU. The other models are likely too large to be trained on a single H100, as they produced out-of-memory errors even with very small batch sizes. Therefore, test predictions are provided only for `CBraMod`.

## Setup

Large files are not stored directly in this repository. They are available in the following Google Drive folder:

[emotion-classification-models](https://drive.google.com/drive/folders/1HVcNsKlzagZ7cHQxvlvWg6poHp03Vbgn?usp=sharing)

After cloning the repository, download the required files and place them in the expected locations.

### Required Files

Before running the project, copy the required data and pretrained model files into the expected locations.

#### Dataset

Copy:

```text
emotion-classification-models/training_test_data/data.npz
```
to:

```
emotion-detection/assets/data.npz
```

#### UniShape checkpoint

Copy:

```
emotion-classification-models/base_models/unishape_checkpoint_finetune.pth
```

to:

```
emotion-detection/code/UniShape/pretrained_model_ckpt/unishape_checkpoint_finetune.pth
```

#### CBraMod pretrained weights

Copy:

```
emotion-classification-models/base_models/pretrained_weights.pth
```

to:

```
emotion-detection/code/CBraMod/pretrained_weights/pretrained_weights.pth
```

The resulting structure should look like this:

```text
emotion-detection/
├── assets/
│   └── data.npz
└── code/
    ├── main_notebook.ipynb
    ├── UniShape/
    │   └── pretrained_model_ckpt/
    │       └── unishape_checkpoint_finetune.pth
    └── CBraMod/
        └── pretrained_weights/
            └── pretrained_weights.pth
```
### Running Inference with Fine-Tuned Models

To run inference using the fine-tuned models without re-running hyperparameter optimization or training, download the trained model folders from:

```text
emotion-classification-models/trained_models/
```

Then place the folders inside:

```text
emotion-detection/results/models/
```

The resulting structure should look like this:
```text
emotion-detection/
└── results/
    └── models/
        ├── cbramod/
        ├── reve/
        └── unishape/
```

## Running on Google Colab

The notebook can be run directly in Google Colab with H100

Open the notebook using the following link:

[Open in Google Colab](https://colab.research.google.com/github/AS-L-C/emotion-detection/blob/main/code/main_notebook.ipynb)

Then add and run the following setup cell at the beginning of the notebook:

```python
%cd /content
!git clone https://github.com/AS-L-C/emotion-detection.git
%cd /content/emotion-detection/code/
```

After running the setup cell, insert your HF token in the second cell (hf_token="") and run the rest of the notebook:

```text
main_notebook.ipynb
```

## Notes

- Large model checkpoints and dataset files are stored externally due to file size constraints.
- Model-specific preprocessing follows the procedures described in the corresponding papers.
- Hyperparameter optimization with Optuna is optional and can be skipped if pretrained or previously selected configurations are used.

## References

- **REVE**: [https://arxiv.org/pdf/2510.21585](https://arxiv.org/pdf/2510.21585)
- **CBraMod**: [https://arxiv.org/pdf/2412.07236](https://arxiv.org/pdf/2412.07236)
- **UniShape**: [https://arxiv.org/pdf/2601.06429](https://arxiv.org/pdf/2601.06429)
