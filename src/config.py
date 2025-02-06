"""
Configuration Settings Module
---------------------------
This module contains all configuration settings for the Anomaly Detection System.
It includes path configurations, model parameters, and logging setup.

Author: [Your Name]
Date: February 2025
"""

import os
import logging
from datetime import datetime

# Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Data configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2
CROSS_VAL_FOLDS = 5

# Feature configurations
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']
COLUMNS_TO_DROP = ['level', 'num_outbound_cmds']

# Model configurations
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': RANDOM_STATE
    },
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'random_state': RANDOM_STATE
    },
    'isolation_forest': {
        'n_estimators': 100,
        'contamination': 'auto',
        'random_state': RANDOM_STATE
    }
}

def setup_logging():
    """
    Configure logging settings for the application.
    Creates a new log file for each run with timestamp.
    """
    # Create timestamp for log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(LOGS_DIR, f'anomaly_detection_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Logging initialized")
