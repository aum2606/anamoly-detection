# Network Anomaly Detection System

A machine learning-based system for detecting network anomalies using the NSL-KDD dataset. This project implements various machine learning models to identify potential network intrusions and security threats.

## Project Structure

```
anomaly_detection/
├── data/                   # Data directory
│   ├── raw/               # Original NSL-KDD dataset
│   └── processed/         # Preprocessed data files
├── src/                   # Source code
│   ├── preprocessing/     # Data preprocessing modules
│   │   ├── __init__.py
│   │   └── preprocess.py # Data preprocessing functions
│   ├── evaluation/       # Model evaluation code
│   │   ├── __init__.py
│   │   └── model_evaluation.py
│   ├── models/          # Trained model files
│   ├── __init__.py
│   └── config.py        # Configuration settings
├── results/             # Evaluation results and visualizations
├── logs/               # Application logs
├── main.py            # Main entry point
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Features

- Data preprocessing pipeline for the NSL-KDD dataset
- Support for multiple machine learning models
- Comprehensive model evaluation metrics:
  - Confusion matrices
  - ROC curves
  - Precision-Recall curves
  - Cross-validation analysis
- Feature importance visualization
- Model performance comparison
- Detailed logging system

## Prerequisites

- Python 3.8 or higher
- Required packages listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd anomaly-detection
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place the NSL-KDD dataset files in the `data/raw` directory:
   - KDDTrain+.txt
   - KDDTest+.txt

2. Run the complete pipeline:
```bash
python main.py
```

Additional command-line options:
- `--skip-preprocessing`: Skip data preprocessing if already done
- `--eval-only`: Run only model evaluation

## Project Components

### 1. Data Preprocessing (`src/preprocessing/preprocess.py`)
- Loads and cleans the NSL-KDD dataset
- Handles missing values
- Encodes categorical features
- Normalizes numerical features
- Converts attack types to binary classification

### 2. Model Evaluation (`src/evaluation/model_evaluation.py`)
- Evaluates model performance
- Generates visualization plots
- Performs cross-validation
- Creates model comparisons
- Analyzes feature importance

### 3. Configuration (`src/config.py`)
- Centralizes project settings
- Defines file paths
- Sets model parameters
- Configures logging

### 4. Main Script (`main.py`)
- Orchestrates the complete workflow
- Handles command-line arguments
- Manages error handling and logging

## Results

The system generates various visualization outputs in the `results` directory:
- Confusion matrices for each model
- ROC curves with AUC scores
- Precision-Recall curves
- Feature importance plots
- Cross-validation comparison plots
- Learning curves for neural networks

## Logging

- Detailed logs are stored in the `logs` directory
- Each run creates a timestamped log file
- Logs include both console output and file logging

## Dependencies

Main dependencies include:
- numpy (≥1.21.0)
- pandas (≥1.3.0)
- scikit-learn (≥0.24.2)
- tensorflow (≥2.6.0)
- matplotlib (≥3.4.3)
- seaborn (≥0.11.2)
- joblib (≥1.0.1)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Add your license information here]

## Authors

[Add author information here]

## Acknowledgments

- NSL-KDD dataset providers
- [Add other acknowledgments]
