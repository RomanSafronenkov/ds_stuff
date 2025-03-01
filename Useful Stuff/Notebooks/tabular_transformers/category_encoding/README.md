# Tabular Data Classification with Category Embeddings

This project demonstrates how to classify tabular data using a neural network with category embeddings for categorical features. The model is implemented in PyTorch and includes features like early stopping, logging, and automatic embedding dimension calculation.

## Project Structure

```
project/
│
├── data/ # Directory for datasets
│ ├── train.csv # Training data
│ ├── val.csv # Validation data
│ └── test.csv # Test data
│
├── model.py # Model and dataset classes
├── train.py # Training script
├── config.py # Configuration file
├── utils.py # Utility functions
├── best_model.pth # Saved best model weights
├── training.log # Training logs
├── tabular_transformers_cat_enc.ipynb # Jupyter notebook with step-by-step explanation
└── README.md # This file
```

## Requirements

To run this project, you need the following Python packages:

- `torch`
- `pandas`
- `numpy`
- `scikit-learn`
- `tabulate`

You can install the dependencies using:

```bash
pip install torch pandas numpy scikit-learn tabulate
```

# How to Run
## 1. Prepare Your Data
Ensure your data is in CSV format and placed in the data/ directory. The data should include:

Training data: `train.csv`

Validation data: `val.csv`

Test data: `test.csv`

## 2. Configure the Project
Edit the config.py file to specify your dataset paths, categorical and numerical columns, and other parameters:

```python
config = {
    "train_data_path": "data/train.csv",
    "val_data_path": "data/val.csv",
    "test_data_path": "data/test.csv",
    "cat_cols": ["cat_feature_1", "cat_feature_2"],  # Categorical columns
    "num_cols": ["num_feature_1", "num_feature_2"],  # Numerical columns
    "target": "target",  # Target column
    "batch_size": 256,
    "lr": 1e-3,
    "n_epochs": 100,
    "patience": 5,
    "best_model_path": "best_model.pth"
}
```

## 3. Train the Model
Run the training script:

```bash
python train.py
```

This will:

- Train the model on the training data.

- Validate the model on the validation data.

- Save the best model weights to best_model.pth.

- Log training progress to training.log.

- Evaluate the model on the test data and print final metrics.

## 4. View Results
After training, you can find:

The best model weights in `best_model.pth.`

Training logs in `training.log.`

Final metrics (loss and accuracy) for train, validation, and test datasets in the logs.

Example output:

```bash
+-----------+-------+----------+
|  Dataset  |  Loss | Accuracy |
+-----------+-------+----------+
|   Train   | 0.400 |   0.82   |
| Validation| 0.480 |   0.79   |
|    Test   | 0.490 |   0.78   |
+-----------+-------+----------+
```

## 5. Explore the Notebook
For a step-by-step explanation of the code, check out the Jupyter notebook:

`tabular_transformers_cat_enc.ipynb`

The notebook covers:

- Data preparation.

- Model architecture.

- Training and evaluation.

- Visualization of results.