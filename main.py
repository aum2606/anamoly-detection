"""
Anomaly Detection System - Main Entry Point
-----------------------------------------
This script serves as the main entry point for the Anomaly Detection System.
It orchestrates the entire workflow from data preprocessing to model evaluation.

Usage:
    python main.py [--skip-preprocessing] [--eval-only]

Author: [Your Name]
Date: February 2025
"""

import os
import argparse
import logging
import pandas as pd
from src.preprocessing.preprocess import load_nsl_kdd_data, preprocess_data
from src.evaluation.model_evaluation import evaluate_all_models
from src.config import (
    setup_logging,
    DATA_DIR, 
    PROCESSED_DATA_DIR, 
    RAW_DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR
)

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, MODELS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Anomaly Detection System')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing step if data is already processed')
    parser.add_argument('--eval-only', action='store_true',
                       help='Run only model evaluation')
    return parser.parse_args()

def run_preprocessing():
    """
    Execute the data preprocessing pipeline.
    
    Returns:
        tuple: Preprocessed data (X_train, X_test, y_train, y_test)
    """
    logging.info("Starting data preprocessing...")
    try:
        # Load data
        train_path = os.path.join(RAW_DATA_DIR, "KDDTrain+.txt")
        test_path = os.path.join(RAW_DATA_DIR, "KDDTest+.txt")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logging.error("Raw data files not found. Please place KDDTrain+.txt and KDDTest+.txt in the data/raw directory.")
            return None
        
        train_data, test_data = load_nsl_kdd_data(train_path, test_path)
        
        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(train_data, test_data)
        
        # Save preprocessed data
        X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "preprocessed_X_train.csv"), index=False)
        X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, "preprocessed_X_test.csv"), index=False)
        y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "preprocessed_y_train.csv"), index=False)
        y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, "preprocessed_y_test.csv"), index=False)
        
        logging.info("Data preprocessing completed successfully")
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        return None

def load_preprocessed_data():
    """
    Load preprocessed data from CSV files.
    
    Returns:
        tuple: Loaded preprocessed data (X_train, X_test, y_train, y_test)
    """
    try:
        X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "preprocessed_X_train.csv"))
        X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "preprocessed_X_test.csv"))
        y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "preprocessed_y_train.csv"))
        y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "preprocessed_y_test.csv"))
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error loading preprocessed data: {str(e)}")
        return None

def main():
    """Main execution function."""
    # Setup logging
    setup_logging()
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup directory structure
    setup_directories()
    
    # Data Processing
    if not args.eval_only and not args.skip_preprocessing:
        data = run_preprocessing()
        if data is None:
            return
        X_train, X_test, y_train, y_test = data
    else:
        # Load preprocessed data
        data = load_preprocessed_data()
        if data is None:
            return
        X_train, X_test, y_train, y_test = data
    
    # Model Evaluation
    logging.info("Starting model evaluation...")
    try:
        evaluate_all_models(X_train, X_test, y_train, y_test)
        logging.info("Model evaluation completed successfully")
    except Exception as e:
        logging.error(f"Error during model evaluation: {str(e)}")

if __name__ == "__main__":
    main()
