# Digit Recognizer: CNN in TensorFlow/Keras

A convolutional neural network for the Kaggle [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition (MNIST handwritten digits). The model classifies 28x28 grayscale images into the digits 0 to 9.

**Public leaderboard score:** 0.98528 (AUC)


## Overview

The script covers the full workflow in a single file:

1. Data loading and quick quality checks (shape, missing values, descriptive statistics)
2. Stratified train / validation / hold-out split
3. Pixel normalization and reshaping into image tensors
4. CNN training with Keras
5. Multi-class ROC AUC evaluation on all three splits
6. Submission file generation

## Approach

### Data splitting

Both splits are stratified on `label` (`random_state=42`) so every digit keeps the same proportion in each subset.

| Split | Share of labeled data | Purpose |
|-------|-----------------------|---------|
| Train | ~70% | Fit the network |
| Validation | ~10% | Monitor training after each epoch |
| Hold-out | 20% | Final unbiased evaluation |

### Preprocessing

- Each 784-pixel row is reshaped to a `28 x 28 x 1` image.
- Pixel values are divided by the maximum pixel value in the train split (255), scaling inputs to `[0, 1]`. The same constant is applied to the validation, hold-out, and Kaggle test sets.

### Model

| Layer | Configuration |
|-------|---------------|
| Conv2D | 32 filters, 3x3, ReLU |
| MaxPooling2D | 2x2 |
| Conv2D | 64 filters, 3x3, ReLU |
| MaxPooling2D | 2x2 |
| Flatten | |
| Dense | 64 units, ReLU |
| Dense | 10 units, linear (logits) |

Training settings:

- Loss: `SparseCategoricalCrossentropy(from_logits=True)`
- Optimizer: Adam, learning rate `0.0002`
- 20 epochs, batch size 32
- Seeds fixed with `tf.random.set_seed(42)` and `np.random.seed(42)`

### Evaluation

Logits are converted to probabilities with softmax, and the macro-averaged one-vs-rest ROC AUC is reported for the train, validation, and hold-out sets. Final predictions are the `argmax` of the class probabilities on the Kaggle test set.

## Getting started

### Prerequisites

- Python 3.9+

```bash
pip install numpy pandas scikit-learn tensorflow
```

### Data

Download `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/competitions/digit-recognizer/data). The data is not included in this repository.

> [!IMPORTANT]
> The script currently reads and writes files using absolute Windows paths. Update the paths in `pd.read_csv(...)` and `to_csv(...)` in `digit recog.py` to match your machine before running.

### Run

```bash
python "digit recog.py"
```

The script trains the model, prints the AUC scores, and writes `kaggle_submission_digits.csv` with the columns `ImageId` and `Label`, ready to upload to Kaggle.

## Project structure

```text
.
├── digit recog.py   # Data loading, preprocessing, CNN training, evaluation, submission
└── README.md
```

## Next steps

- Data augmentation (small rotations, shifts, zooms) to improve generalization
- Batch normalization and dropout for regularization
- Learning-rate scheduling and early stopping
- Ensembling several networks trained with different seeds
