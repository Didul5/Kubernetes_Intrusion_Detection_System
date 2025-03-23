import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class K8sIntrusionDetector:
    def __init__(self, data_path=None, model_path=None):
        """
        Initialize the K8s Intrusion Detection System
        
        Args:
            data_path: Path to the ML-ready dataset
            model_path: Path to save/load trained model
        """
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, data_path=None):
        """
        Load the dataset from specified path
        """
        if data_path:
            self.data_path = data_path
            
        print(f"Loading data from {self.data_path}")
        try:
            # Try to load the already processed ML-ready data
            if 'ml_ready' in self.data_path:
                df = pd.read_csv(self.data_path)
                print(f"Loaded ML-ready dataset with shape: {df.shape}")
                return df
            
            # If loading raw network flows data, handle the special labeling requirements
            elif 'net_flows' in self.data_path:
                df = pd.read_csv(self.data_path)
                print(f"Loaded network flows dataset with shape: {df.shape}")
                
                # Apply labeling based on destination port as specified in the documentation
                # Initialize all as normal (0)
                df['label'] = 0
                
                # Apply labels based on destination port rules from the documentation
                if 'dst_port' in df.columns:
                    # DoS Attack 1 (Slowloris)
                    if 'dvwa' in self.data_path.lower():
                        df.loc[df['dst_port'] == 30025, 'label'] = 1
                    else:  # BoA dataset
                        df.loc[df['dst_port'] == 30026, 'label'] = 1
                    
                    # DoS Attack 2 (Torshammer)
                    df.loc[df['dst_port'] == 30026, 'label'] = 2
                    
                    # Brute Force Attack
                    df.loc[df['dst_port'] == 30027, 'label'] = 3
                    
                    # SQL Injection Attack
                    df.loc[df['dst_port'] == 30025, 'label'] = 4
                    
                    # Drop the dst_port column as recommended in the documentation
                    print("Dropping dst_port column as it was used for labeling")
                    df = df.drop('dst_port', axis=1)
                else:
                    print("Warning: dst_port column not found for labeling")
                
                return df
            
            # If loading merged data
            else:
                df = pd.read_csv(self.data_path)
                print(f"Loaded dataset with shape: {df.shape}")
                return df
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, df):
        """
        Preprocess the data for training:
        - Handle missing values
        - Convert categorical features
        - Scale numerical features
        """
        print("Preprocessing data...")
        
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Check for and handle missing values
        missing_values = data.isna().sum().sum()
        if missing_values > 0:
            print(f"Found {missing_values} missing values. Filling with appropriate values...")
            # Fill numerical missing values with median
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if data[col].isna().sum() > 0:
                    data[col] = data[col].fillna(data[col].median())
            
            # Fill categorical missing values with mode
            cat_cols = data.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if data[col].isna().sum() > 0:
                    data[col] = data[col].fillna(data[col].mode()[0])
        
        # Handle categorical features
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            
        # Separate features and target
        if 'label' in data.columns:
            X = data.drop('label', axis=1)
            y = data['label']
        else:
            print("Warning: 'label' column not found. Using the last column as target.")
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
        
        # Handle features that are likely identifiers or timestamps
        cols_to_drop = [col for col in X.columns if 'id' in col.lower() or 
                         'time' in col.lower() or 'timestamp' in col.lower()]
        
        if cols_to_drop:
            print(f"Dropping identifier/timestamp columns: {cols_to_drop}")
            X = X.drop(cols_to_drop, axis=1)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Preprocessed data: X shape={X.shape}, y shape={y.shape}")
        return X_scaled, y, X.columns
    
    def train(self, X, y, model_type="random_forest"):
        """
        Train the intrusion detection model
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to train ("random_forest" or "gradient_boosting")
        """
        print(f"Training {model_type} model...")
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Check class distribution
        class_distribution = pd.Series(y_train).value_counts(normalize=True)
        print("Class distribution in training data:")
        print(class_distribution)
        
        # Apply SMOTE if we have imbalanced classes
        if any(class_distribution < 0.1):  # If any class is less than 10%
            print("Applying SMOTE for handling class imbalance...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print("Class distribution after SMOTE:")
            print(pd.Series(y_train).value_counts(normalize=True))
        
        # Choose and configure the model
        if model_type == "random_forest":
            # Parameters for RandomForest
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            
        elif model_type == "gradient_boosting":
            # Parameters for GradientBoosting
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.05],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            }
            base_model = GradientBoostingClassifier(random_state=42)
            
        else:
            raise ValueError("model_type must be 'random_forest' or 'gradient_boosting'")
        
        # Use GridSearchCV to find the best hyperparameters
        print("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1, verbose=1, scoring='f1_weighted')
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
        
        return self.model
    
    def save_model(self, model_path=None):
        """
        Save the trained model to disk
        """
        if model_path:
            self.model_path = model_path
        
        if not self.model_path:
            self.model_path = "k8s_intrusion_detector_model.joblib"
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else '.', exist_ok=True)
        
        # Save model
        joblib.dump(self.model, self.model_path)
        
        # Save scaler
        scaler_path = os.path.join(os.path.dirname(self.model_path), "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {self.model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path=None):
        """
        Load a trained model from disk
        """
        if model_path:
            self.model_path = model_path
            
        self.model = joblib.load(self.model_path)
        
        # Try to load the scaler
        scaler_path = os.path.join(os.path.dirname(self.model_path), "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        print(f"Model loaded from {self.model_path}")
        return self.model

    def predict(self, X):
        """
        Make predictions with the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Scale the input if it's not already scaled
        if isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray):
            X_scaled = self.scaler.transform(X)
        else:
            raise ValueError("Input must be a pandas DataFrame or numpy array")
        
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance from the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        if not hasattr(self.model, 'feature_importances_'):
            print("This model does not provide feature importances")
            return None
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(len(indices[:20])), importances[indices[:20]], align='center')
        plt.xticks(range(len(indices[:20])), [feature_names[i] for i in indices[:20]], rotation=90)
        plt.tight_layout()
        plt.show()
        
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)


def main():
    """
    Main function to demonstrate the K8s Intrusion Detector
    """
    # Create detector
    detector = K8sIntrusionDetector()
    
    # Paths to dataset files
    dvwa_ml_ready_path = "C:\\Users\\bhavi\\Downloads\\archive\\dvwa_dataset\\processed\\dvwa_dataset_ml_ready.csv"
    boa_ml_ready_path = "C:\\Users\\bhavi\\Downloads\\archive\\boa_dataset\\processed\\boa_dataset_ml_ready_frontend_microservice.csv"
    
    # Choose which dataset to use
    data_path = dvwa_ml_ready_path  # or boa_ml_ready_path
    
    # Load data
    df = detector.load_data(data_path)
    
    if df is not None:
        # Display dataset info
        print("\nDataset info:")
        print(df.info())
        print("\nClass distribution:")
        if 'label' in df.columns:
            print(df['label'].value_counts())
        
        # Preprocess data
        X, y, feature_names = detector.preprocess_data(df)
        
        # Train model
        detector.train(X, y, model_type="random_forest")
        
        # Get feature importance
        importance_df = detector.get_feature_importance(feature_names)
        print("\nTop 10 most important features:")
        print(importance_df.head(10))
        
        # Save model
        detector.save_model("models/k8s_intrusion_detector.joblib")
        
        print("\nTraining complete!")


if __name__ == "__main__":
    main()
