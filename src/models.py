import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import xgboost as xgb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Conv1D, MaxPooling1D, Flatten
import tensorflow as tf
import joblib
import os

class UnsupervisedModels:
    def __init__(self, save_dir='models'):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def train_kmeans(self, X_train, n_clusters=2):
        """Train K-Means clustering model"""
        print("Training K-Means clustering model...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_train)
        joblib.dump(kmeans, f'{self.save_dir}/kmeans_model.joblib')
        return kmeans
    
    def train_dbscan(self, X_train, eps=0.5, min_samples=5):
        """Train DBSCAN clustering model"""
        print("Training DBSCAN clustering model...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X_train)
        joblib.dump(dbscan, f'{self.save_dir}/dbscan_model.joblib')
        return dbscan
    
    def train_isolation_forest(self, X_train, contamination=0.1):
        """Train Isolation Forest model"""
        print("Training Isolation Forest model...")
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        iso_forest.fit(X_train)
        joblib.dump(iso_forest, f'{self.save_dir}/isolation_forest_model.joblib')
        return iso_forest
    
    def train_autoencoder(self, X_train, encoding_dim=32, epochs=50, batch_size=256):
        """Train Autoencoder model"""
        print("Training Autoencoder model...")
        input_dim = X_train.shape[1]
        
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim * 2, activation='relu')(input_layer)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train the model
        history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
        
        # Save the model
        model_save_path = os.path.join(self.save_dir, 'autoencoder_model.h5')
        autoencoder.save(model_save_path)
        
        return autoencoder, history

class SupervisedModels:
    def __init__(self, save_dir='models'):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        print("Training Random Forest classifier...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model, f'{self.save_dir}/random_forest_model.joblib')
        return rf_model
    
    def train_svm(self, X_train, y_train):
        """Train SVM classifier"""
        print("Training SVM classifier...")
        svm_model = SVC(kernel='rbf', random_state=42)
        svm_model.fit(X_train, y_train)
        joblib.dump(svm_model, f'{self.save_dir}/svm_model.joblib')
        return svm_model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost classifier"""
        print("Training XGBoost classifier...")
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        joblib.dump(xgb_model, f'{self.save_dir}/xgboost_model.joblib')
        return xgb_model
    
    def train_cnn(self, X_train, y_train, epochs=50, batch_size=32):
        """Train CNN model"""
        print("Training CNN model...")
        # Reshape input data for CNN (samples, timesteps, features)
        X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Save the model
        model_save_path = os.path.join(self.save_dir, 'cnn_model.h5')
        model.save(model_save_path)
        
        return model, history

    def train_lstm(self, X_train, y_train, epochs=50, batch_size=32):
        """Train LSTM model"""
        print("Training LSTM model...")
        # Reshape input data for LSTM (samples, timesteps, features)
        X_train_reshaped = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
        
        model = Sequential([
            LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=True),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Save the model
        model_save_path = os.path.join(self.save_dir, 'lstm_model.h5')
        model.save(model_save_path)
        
        return model, history

def evaluate_model(model, X_test, y_test, model_type='supervised'):
    """Evaluate model performance"""
    if model_type == 'supervised':
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Model Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
    
    elif model_type == 'unsupervised':
        # For unsupervised models, we'll use different evaluation metrics
        if hasattr(model, 'labels_'):
            # For clustering models (K-means, DBSCAN)
            labels = model.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"Number of clusters: {n_clusters}")
            print(f"Number of samples in each cluster: {np.bincount(labels + 1)}")
        elif hasattr(model, 'predict'):
            # For anomaly detection models (Isolation Forest, Autoencoder)
            predictions = model.predict(X_test)
            if isinstance(predictions, np.ndarray):
                anomaly_ratio = np.mean(predictions == -1)
                print(f"Anomaly ratio: {anomaly_ratio:.4f}")

def main():
    # Load preprocessed data
    X_train = pd.read_csv('preprocessed_X_train.csv')
    X_test = pd.read_csv('preprocessed_X_test.csv')
    y_train = pd.read_csv('preprocessed_y_train.csv')
    y_test = pd.read_csv('preprocessed_y_test.csv')
    
    # Initialize model classes
    unsupervised_models = UnsupervisedModels()
    supervised_models = SupervisedModels()
    
    # Train and evaluate unsupervised models
    print("\n=== Training Unsupervised Models ===")
    
    kmeans_model = unsupervised_models.train_kmeans(X_train)
    dbscan_model = unsupervised_models.train_dbscan(X_train)
    iso_forest_model = unsupervised_models.train_isolation_forest(X_train)
    autoencoder_model, _ = unsupervised_models.train_autoencoder(X_train)
    
    # Train and evaluate supervised models
    print("\n=== Training Supervised Models ===")
    
    rf_model = supervised_models.train_random_forest(X_train, y_train.values.ravel())
    svm_model = supervised_models.train_svm(X_train, y_train.values.ravel())
    xgb_model = supervised_models.train_xgboost(X_train, y_train.values.ravel())
    cnn_model, _ = supervised_models.train_cnn(X_train, y_train.values.ravel())
    lstm_model, _ = supervised_models.train_lstm(X_train, y_train.values.ravel())
    
    print("\n=== Evaluating Models ===")
    from model_evaluation import evaluate_all_models
    
    # Convert y_train and y_test to 1D arrays
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    # Evaluate all models and create visualizations
    evaluator, ensemble_model = evaluate_all_models(X_train, X_test, y_train, y_test)
    
    print("\nEvaluation completed! Check the 'results' directory for visualization plots.")

if __name__ == "__main__":
    main()
