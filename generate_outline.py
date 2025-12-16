"""
generate_outline.py - Inference script to produce JSON outlines
Uses trained ML model to extract structured outlines from PDF files.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from extract_lines import LineExtractor


class OutlineGenerator:
    """Generates structured outlines from PDF files using trained ML model."""
    
    def __init__(self, model_path: str = 'model.pkl', encoder_path: str = 'label_encoder.pkl'):
        """
        Initialize the outline generator.
        
        Args:
            model_path: Path to the trained model file
            encoder_path: Path to the label encoder file
        """
        self.line_extractor = LineExtractor()
        self.model_data = None
        self.label_encoder = None
        self.load_model(model_path, encoder_path)
    
    def load_model(self, model_path: str, encoder_path: str):
        """Load the trained model and label encoder."""
        try:
            self.model_data = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            print("Model and label encoder loaded successfully.")
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            print("Please train the model first using train_model.py")
            raise
    
    def predict_line_labels(self, lines_data: List[Dict]) -> Tuple[List[str], List[float]]:
        """
        Predict labels for extracted lines.
        
        Args:
            lines_data: List of line feature dictionaries
            
        Returns:
            Tuple of (predicted_labels, confidence_scores)
        """
        if not lines_data:
            return [], []
        
        # Prepare features
        df = pd.DataFrame(lines_data)
        feature_columns = self.model_data['feature_columns']
        
        X = df[feature_columns].copy()
        
        # Convert boolean columns to numeric
        bool_columns = ['is_bold', 'is_italic', 'is_title_case', 'is_upper_case', 'has_numbers']
        for col in bool_columns:
            if col in X.columns:
                X[col] = X[col].astype(int)
        
        # Scale features
        X_scaled = self.model_data['scaler'].transform(X)
        
        # Predict
        predictions = self.model_data['model'].predict(X_scaled)
        probabilities = self.model_data['model'].predict_proba(X_scaled)
        
        # Get confidence scores (max probability for each prediction)
        confidence_scores = np.max(probabilities, axis=1)
        
        # Decode labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels.tolist(), confidence_scores.tolist()
    
    def extract_title(self, lines_data: List[Dict], predicted_labels: List[str], 
                     confidence_scores: List[float]) -> str:
        """
        Extract the document title based on predictions.
        
        Args:
            lines_data: List of line feature dictionaries
            predicted_labels: Predicted labels for each line
            confidence_scores: Confidence scores for predictions
            
        Returns:
            Document title string
        """
        title_candidates = []
        
        for i, (line, label, confidence) in enumerate(zip(lines_data, predicted_labels, confidence_scores)):
            if label == 'title' and confidence > 0.5:
                title_candidates.append({
                    'text': line['text'].strip(),
                    'confidence': confidence,
                    'page': line['page'],
                    'font_size': line['font_size_avg']
                })
        
        if title_candidates:
            # Choose title with highest confidence
            best_title = max(title_candidates, key=lambda x: x['confidence'])
            return best_title['text']
        else:
            # Fallback: use the line with largest font size on first page
            first_page_lines = [line for line in lines_data if line['page'] == 1]
            if first_page_lines:
                # Filter out very short texts (likely page numbers or artifacts)
                meaningful_lines = [line for line in first_page_lines if len(line['text'].strip()) > 5]
                if meaningful_lines:
                    largest_font_line = max(meaningful_lines, key=lambda x: x['font_size_avg'])
                    return largest_font_line['text'].strip()
        
        return "Unknown Document"
    
    def extract_outline(self, lines_data: List[Dict], predicted_labels: List[str], 
                       confidence_scores: List[float]) -> List[Dict]:
        """
        Extract the document outline based on predictions.
        
        Args:
            lines_data: List of line feature dictionaries
            predicted_labels: Predicted labels for each line
            confidence_scores: Confidence scores for predictions
            
        Returns:
            List of outline entries
        """
        outline = []
        heading_levels = ['H1', 'H2', 'H3']  # Removed H4
        seen_texts = set()  # To avoid duplicates
        
        for i, (line, label, confidence) in enumerate(zip(lines_data, predicted_labels, confidence_scores)):
            if label in heading_levels and confidence > 0.6:  # Increased confidence threshold
                # Normalize text for duplicate detection
                normalized_text = line['text'].strip().lower()
                
                # Skip if we've already seen this text
                if normalized_text in seen_texts:
                    continue
                
                # Skip very short headings (likely noise)
                if len(line['text'].strip()) < 3:
                    continue
                
                outline_entry = {
                    'level': label,
                    'text': line['text'].strip(),
                    'page': line['page']
                }
                outline.append(outline_entry)
                seen_texts.add(normalized_text)
        
        # Sort by page number and then by y_position (top to bottom)
        outline.sort(key=lambda x: (x['page'], 
                                   next(line['y_position'] for line in lines_data 
                                       if line['text'].strip() == x['text'] and line['page'] == x['page'])))
        
        return outline
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Process a single PDF file and generate outline.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing title and outline
        """
        print(f"Processing {os.path.basename(pdf_path)}...")
        
        # Extract line features
        lines_data = self.line_extractor.extract_lines_from_pdf(pdf_path)
        
        if not lines_data:
            return {
                'title': 'Empty Document',
                'outline': []
            }
        
        # Predict labels
        predicted_labels, confidence_scores = self.predict_line_labels(lines_data)
        
        # Extract title
        title = self.extract_title(lines_data, predicted_labels, confidence_scores)
        
        # Extract outline
        outline = self.extract_outline(lines_data, predicted_labels, confidence_scores)
        
        result = {
            'title': title,
            'outline': outline
        }
        
        print(f"Extracted title: {title}")
        print(f"Found {len(outline)} outline entries")
        
        return result
    
    def process_directory(self, input_dir: str, output_dir: str):
        """
        Process all PDF files in input directory and save results to output directory.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save JSON results
        """
        if not os.path.exists(input_dir):
            print(f"Input directory '{input_dir}' not found.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in '{input_dir}'")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                pdf_path = os.path.join(input_dir, pdf_file)
                
                # Process PDF
                result = self.process_pdf(pdf_path)
                
                # Save result
                output_filename = os.path.splitext(pdf_file)[0] + '.json'
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"Saved outline to {output_filename}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                continue
        
        print(f"Processing completed. Results saved to '{output_dir}'")


def main():
    """Main function to process PDFs and generate outlines."""
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # For local development, use relative paths
    if not os.path.exists(input_dir):
        input_dir = "input"
        output_dir = "output"
    
    # Check if model files exist
    if not os.path.exists('model.pkl') or not os.path.exists('label_encoder.pkl'):
        print("Model files not found. Please train the model first using train_model.py")
        return
    
    # Initialize generator and process files
    try:
        generator = OutlineGenerator()
        generator.process_directory(input_dir, output_dir)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
