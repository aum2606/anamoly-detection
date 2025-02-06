"""
Data Preprocessing Module
------------------------
This module handles the preprocessing of the NSL-KDD dataset for anomaly detection.
It includes functions for loading, cleaning, and transforming the raw data.

The module performs the following preprocessing steps:
1. Data Loading: Load raw NSL-KDD dataset
2. Feature Selection: Remove unnecessary columns
3. Missing Value Handling: Impute missing values
4. Categorical Encoding: Convert categorical features to numeric
5. Feature Scaling: Standardize numeric features

Author: [Your Name]
Date: February 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging
from ..config import CATEGORICAL_FEATURES, COLUMNS_TO_DROP, RANDOM_STATE

def load_nsl_kdd_data(train_path, test_path):
    """
    Load the NSL-KDD dataset from specified paths.
    
    Args:
        train_path (str): Path to the training data file
        test_path (str): Path to the test data file
    
    Returns:
        tuple: (train_data, test_data) as pandas DataFrames
    
    Raises:
        FileNotFoundError: If either data file is not found
        pd.errors.EmptyDataError: If either file is empty
    """
    # Column names for the NSL-KDD dataset
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
              'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
              'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
              'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
              'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
              'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
              'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
              'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
              'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
              'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level']
    
    logging.info("Loading training data...")
    train_data = pd.read_csv(train_path, header=None, names=columns)
    logging.info(f"Training data shape: {train_data.shape}")
    
    logging.info("Loading test data...")
    test_data = pd.read_csv(test_path, header=None, names=columns)
    logging.info(f"Test data shape: {test_data.shape}")
    
    return train_data, test_data

def preprocess_data(train_data, test_data):
    """
    Preprocess the NSL-KDD dataset for anomaly detection.
    
    Args:
        train_data (pd.DataFrame): Raw training data
        test_data (pd.DataFrame): Raw test data
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) preprocessed data
        
    Steps:
        1. Feature Selection
        2. Missing Value Imputation
        3. Categorical Feature Encoding
        4. Feature Scaling
    """
    logging.info("Starting data preprocessing...")
    
    # Create copies to avoid modifying original data
    train_df = train_data.copy()
    test_df = test_data.copy()
    
    # 1. Feature Selection
    logging.info("Removing unnecessary columns...")
    train_df = train_df.drop(COLUMNS_TO_DROP, axis=1)
    test_df = test_df.drop(COLUMNS_TO_DROP, axis=1)
    
    # 2. Handle Missing Values
    logging.info("Handling missing values...")
    numeric_columns = train_df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    
    train_df[numeric_columns] = imputer.fit_transform(train_df[numeric_columns])
    test_df[numeric_columns] = imputer.transform(test_df[numeric_columns])
    
    # 3. Label Encoding for categorical features
    logging.info("Encoding categorical features...")
    label_encoders = {}
    
    for column in CATEGORICAL_FEATURES:
        label_encoders[column] = LabelEncoder()
        train_df[column] = label_encoders[column].fit_transform(train_df[column])
        test_df[column] = label_encoders[column].transform(test_df[column])
    
    # Convert attack labels to binary (normal vs attack)
    logging.info("Converting attack labels to binary...")
    train_df['attack'] = train_df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['attack'] = test_df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # 4. Feature Normalization
    logging.info("Normalizing features...")
    # Separate features and labels
    X_train = train_df.drop('attack', axis=1)
    y_train = train_df['attack']
    X_test = test_df.drop('attack', axis=1)
    y_test = test_df['attack']
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to DataFrame to maintain column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    logging.info("Preprocessing completed successfully")
    logging.info(f"Final shapes - X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def main():
    # Update paths to reflect new directory structure
    train_path = "../../data/raw/KDDTrain+.txt"
    test_path = "../../data/raw/KDDTest+.txt"
    
    print("Loading NSL-KDD dataset...")
    train_data, test_data = load_nsl_kdd_data(train_path, test_path)
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(train_data, test_data)
    
    print("Dataset shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Save preprocessed data
    X_train.to_csv("../../data/processed/preprocessed_X_train.csv", index=False)
    X_test.to_csv("../../data/processed/preprocessed_X_test.csv", index=False)
    y_train.to_csv("../../data/processed/preprocessed_y_train.csv", index=False)
    y_test.to_csv("../../data/processed/preprocessed_y_test.csv", index=False)
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
