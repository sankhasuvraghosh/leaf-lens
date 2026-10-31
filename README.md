# Leaf Species Classification (DRISHTI 1.0)

A PyTorch convolutional neural network that identifies the species of a leaf from a photograph. It is trained on the DRISHTI 1.0 training kit (five species, `sp-01` to `sp-05`) and checked against the two practice sets (`demo_test_1`, `demo_test_2`) using the provided Excel label files.

## Overview

| | |
|---|---|
| Task | 5-class image classification (leaf species) |
| Framework | PyTorch + torchvision |
| Model | Custom 4-block CNN (Conv, BatchNorm, ReLU, MaxPool), global average pooling, small MLP head |
| Input | RGB photos resized to 160 x 160 |
| Evaluation | Held-out validation split, then `demo_test_1` and `demo_test_2` against their label files |

## Project structure

```
.
├──drishti.ipynb       # data loading, training, validation, single-image check, demo prediction scores the trained model on demo_test_1 and demo_test_2
├── leaf_dataset_model.pth      # best model weights (created by training)
├── confusion_matrix.png        # validation confusion matrix (created by training)
├── demo_predictions.csv        # predictions for demo_path (created by training)
├── pred_demo_test_1.csv        # created by test_demo.py
├── pred_demo_test_2.csv        # created by test_demo.py
└── demo_confusion_matrix.png   # created by test_demo.py
```

The dataset folder from the kit is expected to look like this:

```
DRISHTI_1_0_Training_Kit/
└── Deliverables/
    ├── Train/
    │   ├── sp-01/   (leaf photos)
    │   ├── sp-02/
    │   ├── sp-03/
    │   ├── sp-04/
    │   └── sp-05/
    ├── demo_test_1/
    ├── demo_test_2/
    ├── demo_test1_true_label.xlsx
    ├── demo_test2_true_label.xlsx
    └── check_submission.py
```

The dataset itself is not included in this repository.

## Setup

Python 3.9 or newer is recommended. A GPU is optional but makes training much faster.

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas pillow openpyxl
```

`openpyxl` is needed to read the Excel label files.

## How to run

The scripts are written as flat, notebook-style code, so each section can be pasted into its own cell in a Jupyter notebook (`.ipynb`) and run top to bottom in one session.


## Method

**Data split.** The `Train` folder is split 80/20 with stratification (fixed seed 42) so every species appears in both parts.

**Augmentation (training only).** Random resized crop (scale 0.6 to 1.0), horizontal and vertical flips, rotation up to 30 degrees, and mild colour jitter. The colour jitter is kept small because leaf colour is an informative feature.

**Model.** Four convolutional blocks with 32, 64, 128 and 256 channels, each with BatchNorm, ReLU and 2x2 max pooling, followed by global average pooling and a head of `Linear(256, 128)`, ReLU, `Dropout(0.5)` and `Linear(128, 5)`.

**Optimisation.**

| Setting | Value |
|---|---|
| Loss | Cross-entropy with label smoothing 0.1 |
| Optimizer | Adam, learning rate 1e-3, weight decay 1e-4 |
| Scheduler | StepLR, halve the learning rate every 10 epochs |
| Epochs | 40 |
| Batch size | 32 |
| Checkpointing | Best validation accuracy, not the last epoch |

**Inference.** `test_demo.py` averages the predictions for the original image, a horizontal flip and a vertical flip.

## Results

Fill these in after running.

| Set | Images | Accuracy |
|---|---|---|
| Validation split | _ | 89 % |
| demo_test_1 | _ |100 % |
| demo_test_2 | _ | 90 % |

Note: if the training folders contain several photos of the same leaf, near-duplicates can land in both the training and validation parts and make the validation score look better than it is. The demo sets are the more honest measure.



## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` for a demo folder or label file | Check that `base` points at the folder containing them, and that the label file names match the `demos` list in `test_demo.py` |
| `ImportError: openpyxl` | `pip install openpyxl`, then restart the kernel |
| "matched with labels" is lower than "images predicted" | Image names in the Excel file differ from the files. The script prints the columns and first rows of each label file. It assumes column 1 is the image name and column 2 is the species |
| Training is very slow | Check `print(device)`. On CPU, lower `epochs` to about 15 and `img_size` to 128 for a trial run |
| `NameError: model is not defined` | Run the training cells first, or restart and re-run the cells that define the model and transforms |

## Ideas for improvement

- Switch to a pretrained backbone (for example ResNet18 with a new final layer). This usually gives the largest accuracy gain on small leaf datasets; use a lower learning rate such as 3e-4.
- Increase `img_size` to 224 or 256 so fine details such as leaf margins and veins are preserved.
- Retrain on all images (no validation split) once the settings are final.
- Inspect the misclassified table and confusion matrix to see which species pairs are confused, then add targeted augmentation or more data for them.
