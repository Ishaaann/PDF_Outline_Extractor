"""
improved_train_model.py - Enhanced ML model training
Improved version with hyperparameter tuning and ensemble methods.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import os
from improved_extract_lines import ImprovedLineExtractor


class ImprovedOutlineModelTrainer:
    """Enhanced ML model trainer with better feature engineering and model selection."""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.feature_columns = [
            'font_size_avg_norm',
            'font_size_max_norm',
            'font_size_min',
            'font_size_variance',
            'font_size_relative',
            'font_size_percentile',
            'is_bold',
            'is_italic',
            'x_position_norm',
            'y_position_norm',
            'line_width_norm',
            'word_count',
            'char_length',
            'is_title_case',
            'is_upper_case',
            'has_numbers',
            'starts_with_number',
            'is_centered',
            'has_colon',
            'ends_with_period',
            'all_caps_words',
            'punctuation_ratio',
            'font_consistency',
            'indentation_level',
            'prev_font_size',
            'next_font_size',
            'font_size_diff_prev',
            'font_size_diff_next',
            'relative_position_in_page'
        ]
    
    def load_training_data(self, csv_path: str) -> tuple:
        """Load and prepare enhanced training data."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training data file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Check if we have labeled data
        if 'label' not in df.columns:
            raise ValueError("CSV file must contain a 'label' column")
        
        # Filter out None labels and H4 labels for training but keep some for negative examples
        df_filtered = df[(df['label'] != 'None') & (df['label'] != 'H4')].copy()
        
        # Add some 'None' samples to help with classification
        none_samples = df[df['label'] == 'None'].sample(n=min(200, len(df[df['label'] == 'None'])), random_state=42)
        df_filtered = pd.concat([df_filtered, none_samples], ignore_index=True)
        
        print(f"Loaded {len(df)} total training examples")
        print(f"Using {len(df_filtered)} examples for training")
        print(f"Label distribution:\n{df_filtered['label'].value_counts()}")
        
        # Handle missing columns by using available features
        available_features = [col for col in self.feature_columns if col in df_filtered.columns]
        missing_features = [col for col in self.feature_columns if col not in df_filtered.columns]
        
        if missing_features:
            print(f"Missing features (will be ignored): {missing_features}")
        
        self.feature_columns = available_features
        print(f"Using {len(self.feature_columns)} features for training")
        
        # Prepare features
        X = df_filtered[self.feature_columns].copy()
        
        # Convert boolean columns to numeric
        bool_columns = [col for col in X.columns if df_filtered[col].dtype == 'bool' or col.startswith('is_') or col.startswith('has_') or col.startswith('ends_')]
        for col in bool_columns:
            X[col] = X[col].astype(int)
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        # Prepare labels
        y = df_filtered['label'].values
        
        return X, y
    
    def train_model_with_tuning(self, X, y):
        """Train model with hyperparameter tuning."""
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
        
        print("Testing different models with hyperparameter tuning...")
        
        # Test Random Forest with hyperparameter tuning
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']
        }
        
        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='f1_weighted', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        
        print(f"Best Random Forest params: {rf_grid.best_params_}")
        print(f"Best Random Forest CV score: {rf_grid.best_score_:.3f}")
        
        # Test Gradient Boosting
        gb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        gb = GradientBoostingClassifier(random_state=42)
        gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='f1_weighted', n_jobs=-1)
        gb_grid.fit(X_train, y_train)
        
        print(f"Best Gradient Boosting params: {gb_grid.best_params_}")
        print(f"Best Gradient Boosting CV score: {gb_grid.best_score_:.3f}")
        
        # Choose best model
        if rf_grid.best_score_ > gb_grid.best_score_:
            self.model = rf_grid.best_estimator_
            best_model_name = "Random Forest"
            best_score = rf_grid.best_score_
        else:
            self.model = gb_grid.best_estimator_
            best_model_name = "Gradient Boosting"
            best_score = gb_grid.best_score_
        
        print(f"\nSelected model: {best_model_name} (CV F1: {best_score:.3f})")
        
        # Evaluate on test set
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"Test F1-score: {f1:.3f}")
        
        try:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        except Exception as e:
            print(f"Could not generate classification report: {e}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 15 Most Important Features:")
            print(feature_importance.head(15))
    
    def save_model(self, model_path: str = 'improved_model.pkl', encoder_path: str = 'improved_label_encoder.pkl'):
        """Save the trained model and encoders."""
        if self.model is None or self.label_encoder is None:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"Model saved to {model_path}")
        print(f"Label encoder saved to {encoder_path}")


def regenerate_training_data():
    """Regenerate training data with improved features."""
    print("Regenerating training data with improved features...")
    
    from create_training_data import TrainingDataGenerator
    
    # Create a modified version that uses improved extractor
    class ImprovedTrainingDataGenerator(TrainingDataGenerator):
        def __init__(self):
            self.line_extractor = ImprovedLineExtractor()
    
    generator = ImprovedTrainingDataGenerator()
    result = generator.generate_training_data("input", "data", "improved_training_data.csv")
    return result


def main():
    """Main function to train the improved model."""
    # Regenerate training data with improved features
    training_csv = regenerate_training_data()
    
    if not training_csv:
        print("Failed to generate training data")
        return
    
    # Train improved model
    trainer = ImprovedOutlineModelTrainer()
    
    try:
        # Load data
        X, y = trainer.load_training_data(training_csv)
        
        # Train with hyperparameter tuning
        trainer.train_model_with_tuning(X, y)
        
        # Save model
        trainer.save_model('improved_model.pkl', 'improved_label_encoder.pkl')
        
        print("\n✅ Improved model training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
