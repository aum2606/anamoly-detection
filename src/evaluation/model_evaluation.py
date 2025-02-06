"""
Model Evaluation Module
----------------------
This module provides comprehensive evaluation capabilities for machine learning models
used in anomaly detection. It includes visualization, metrics calculation, and
model comparison functionality.

The module performs the following evaluations:
1. Model Performance Metrics
   - Confusion Matrix
   - ROC Curve and AUC
   - Precision-Recall Curve
2. Cross-validation Analysis
3. Feature Importance Analysis
4. Learning Curves Visualization
5. Model Comparison

Author: [Your Name]
Date: February 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.ensemble import VotingClassifier
import joblib
import os
import logging
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.cluster import DBSCAN
from ..config import RESULTS_DIR, MODELS_DIR, CROSS_VAL_FOLDS, RANDOM_STATE

class ModelEvaluator:
    """
    A class for comprehensive model evaluation and visualization.
    
    This class provides methods for:
    - Performance visualization (ROC curves, confusion matrices)
    - Cross-validation analysis
    - Model comparison
    - Feature importance analysis
    - Learning curves visualization
    """
    
    def __init__(self, results_dir=RESULTS_DIR):
        """
        Initialize the ModelEvaluator.
        
        Args:
            results_dir (str): Directory to save evaluation results
        """
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Store evaluation results
        self.model_scores = {}
        self.cross_val_results = {}
        logging.info(f"ModelEvaluator initialized with results directory: {results_dir}")
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """
        Plot and save confusion matrix for a model.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            model_name (str): Name of the model
        """
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{self.results_dir}/{model_name}_confusion_matrix.png')
        plt.close()
        logging.info(f"Confusion matrix saved for {model_name}")
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name):
        """
        Plot and save ROC curve for a model.
        
        Args:
            y_true (array-like): True labels
            y_pred_proba (array-like): Predicted probabilities
            model_name (str): Name of the model
            
        Returns:
            float: Area under the ROC curve
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f'{self.results_dir}/{model_name}_roc_curve.png')
        plt.close()
        
        logging.info(f"ROC curve saved for {model_name} (AUC: {roc_auc:.4f})")
        return roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name):
        """Plot Precision-Recall curve for a model"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.savefig(f'{self.results_dir}/{model_name}_pr_curve.png')
        plt.close()
        
        logging.info(f"Precision-Recall curve saved for {model_name} (AUC: {pr_auc:.4f})")
        return pr_auc
    
    def perform_cross_validation(self, model, X, y, model_name, n_splits=CROSS_VAL_FOLDS):
        """Perform cross-validation and store results"""
        print(f"\nPerforming cross-validation for {model_name}...")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        
        # Get cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
        self.cross_val_results[model_name] = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'all_scores': cv_scores
        }
        
        print(f"Cross-validation results for {model_name}:")
        print(f"Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        logging.info(f"Cross-validation completed for {model_name} (mean accuracy: {cv_scores.mean():.4f})")
        return cv_scores
    
    def plot_cross_validation_comparison(self):
        """Plot comparison of cross-validation results"""
        plt.figure(figsize=(12, 6))
        
        models = list(self.cross_val_results.keys())
        mean_scores = [self.cross_val_results[model]['mean_score'] for model in models]
        std_scores = [self.cross_val_results[model]['std_score'] for model in models]
        
        bars = plt.bar(models, mean_scores, yerr=std_scores, capsize=5)
        plt.title('Cross-validation Results Comparison')
        plt.xlabel('Models')
        plt.ylabel('Mean Accuracy Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.savefig(f'{self.results_dir}/cross_validation_comparison.png')
        plt.close()
        
        logging.info("Cross-validation comparison plot saved")
    
    def create_ensemble_model(self, models_dict, X_train, y_train):
        """Create and train an ensemble model using voting classifier"""
        print("\nCreating ensemble model...")
        
        # Prepare models for voting classifier
        estimators = []
        for name, model in models_dict.items():
            if hasattr(model, 'predict_proba'):  # Check if model supports probability predictions
                estimators.append((name, model))
        
        # Create voting classifier
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # Save the ensemble model
        joblib.dump(ensemble, f'{self.results_dir}/ensemble_model.joblib')
        
        logging.info("Ensemble model created and saved")
        return ensemble
    
    def plot_feature_importance(self, model, feature_names, model_name):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.title(f'Feature Importances ({model_name})')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/{model_name}_feature_importance.png')
            plt.close()
            
            logging.info(f"Feature importance plot saved for {model_name}")
    
    def plot_learning_curves(self, history, model_name):
        """Plot learning curves for deep learning models"""
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Model accuracy ({model_name})')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model loss ({model_name})')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/{model_name}_learning_curves.png')
        plt.close()
        
        logging.info(f"Learning curves plot saved for {model_name}")

def evaluate_all_models(X_train, X_test, y_train, y_test, models_dir=MODELS_DIR):
    """
    Evaluate all trained models and create visualizations.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training labels
        y_test (pd.Series): Test labels
        models_dir (str): Directory containing saved models
    """
    evaluator = ModelEvaluator()
    
    # Load all saved models
    models = {}
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        
        try:
            if model_name.endswith('.joblib'):
                # Load scikit-learn models
                model = joblib.load(model_path)
                name = model_name.split('.')[0]
                models[name] = model
                logging.info(f"Loaded scikit-learn model: {name}")
            elif model_name.endswith('.h5'):
                # Load TensorFlow models
                model = load_model(model_path)
                name = model_name.split('.')[0]
                models[name] = model
                logging.info(f"Loaded TensorFlow model: {name}")
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {str(e)}")
    
    logging.info(f"Loaded {len(models)} models for evaluation")
    
    # Evaluate each model
    for name, model in models.items():
        logging.info(f"\nEvaluating {name}...")
        
        try:
            # Special handling for different model types
            if isinstance(model, DBSCAN):
                # Handle unsupervised models differently
                predictions = model.fit_predict(X_test)
                evaluator.plot_confusion_matrix(y_test, predictions, name)
            else:
                # Standard evaluation for supervised models
                predictions = model.predict(X_test)
                evaluator.plot_confusion_matrix(y_test, predictions, name)
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_test)[:, 1]
                    evaluator.plot_roc_curve(y_test, probabilities, name)
                    evaluator.plot_precision_recall_curve(y_test, probabilities, name)
                
                # Perform cross-validation
                cv_scores = evaluator.perform_cross_validation(model, X_train, y_train, name)
                logging.info(f"Cross-validation scores for {name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            # Plot feature importance if available
            if hasattr(model, 'feature_importances_'):
                evaluator.plot_feature_importance(model, X_train.columns, name)
            
            # Plot learning curves for neural networks
            if hasattr(model, 'history'):
                evaluator.plot_learning_curves(model.history, name)
                
        except Exception as e:
            logging.error(f"Error evaluating model {name}: {str(e)}")
    
    # Plot cross-validation comparison
    evaluator.plot_cross_validation_comparison()
    logging.info("Model evaluation completed")
    
    # Create and evaluate ensemble model (excluding neural networks and clustering models)
    ensemble_model = evaluator.create_ensemble_model(
        {name: model for name, model in models.items() 
         if not isinstance(model, (tf.keras.Model, DBSCAN)) and hasattr(model, 'predict_proba')},
        X_train, y_train
    )
    
    return evaluator, ensemble_model
