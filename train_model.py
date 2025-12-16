"""
train_model.py - Trains ML model for PDF outline extraction
Trains a supervised ML model using manually labeled line data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os


class OutlineModelTrainer:
    """Trains and evaluates ML models for PDF outline extraction."""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.feature_columns = [
            'font_size_avg_norm',
            'font_size_max_norm', 
            'is_bold',
            'is_italic',
            'x_position_norm',
            'y_position_norm',
            'word_count',
            'char_length',
            'is_title_case',
            'is_upper_case',
            'has_numbers',
            'font_size_relative'
        ]
    
    def load_training_data(self, csv_path: str) -> tuple:
        """
        Load and prepare training data from CSV.
        
        Args:
            csv_path: Path to the labeled CSV file
            
        Returns:
            Tuple of (X, y) - features and labels
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training data file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Check if we have labeled data
        if 'label' not in df.columns:
            raise ValueError("CSV file must contain a 'label' column")
        
        # Remove rows with 'None' labels for training (optional)
        df_filtered = df[df['label'] != 'None']  # Remove None labels for better training
        
        print(f"Loaded {len(df)} training examples")
        print(f"After filtering 'None' labels: {len(df_filtered)} examples")
        print(f"Label distribution:\n{df_filtered['label'].value_counts()}")
        
        # Prepare features
        X = df_filtered[self.feature_columns].copy()
        
        # Convert boolean columns to numeric
        bool_columns = ['is_bold', 'is_italic', 'is_title_case', 'is_upper_case', 'has_numbers']
        for col in bool_columns:
            X[col] = X[col].astype(int)
        
        # Prepare labels
        y = df_filtered['label'].values
        
        return X, y
    
    def train_model(self, X, y, model_type: str = 'random_forest'):
        """
        Train the classification model.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model ('random_forest', 'logistic_regression')
        """
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        print(f"Training {model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        try:
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        except Exception as e:
            print(f"Could not generate classification report: {e}")
            # Alternative: just show accuracy per class
            from sklearn.metrics import accuracy_score
            print(f"Overall accuracy: {accuracy_score(y_test, y_pred):.3f}")
        
        # Feature importance (for Random Forest)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nFeature Importance:")
            print(feature_importance)
    
    def save_model(self, model_path: str = 'model.pkl', encoder_path: str = 'label_encoder.pkl'):
        """
        Save the trained model and label encoder.
        
        Args:
            model_path: Path to save the model
            encoder_path: Path to save the label encoder
        """
        if self.model is None or self.label_encoder is None:
            raise ValueError("Model must be trained before saving")
        
        # Save model, scaler, and label encoder
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"Model saved to {model_path}")
        print(f"Label encoder saved to {encoder_path}")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels and probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)
        
        # Decode labels
        labels = self.label_encoder.inverse_transform(y_pred)
        
        return labels, y_proba


def main():
    """Main function to train the model."""
    # Check if training data exists
    training_csv = "training_data.csv"
    
    if not os.path.exists(training_csv):
        print(f"Training data file '{training_csv}' not found.")
        print("Please run create_training_data.py first to generate labeled training data.")
        return
    
    # Train model
    trainer = OutlineModelTrainer()
    
    try:
        # Load data
        X, y = trainer.load_training_data(training_csv)
        
        # Train Random Forest model
        trainer.train_model(X, y, model_type='random_forest')
        
        # Save model
        trainer.save_model('model.pkl', 'label_encoder.pkl')
        
        print("\nModel training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    main()
