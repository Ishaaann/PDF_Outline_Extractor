"""
improved_generate_outline.py - Enhanced inference script
Improved version with better title extraction and outline generation.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from improved_extract_lines import ImprovedLineExtractor
import re


class ImprovedOutlineGenerator:
    """Enhanced outline generator with improved accuracy."""
    
    def __init__(self, model_path: str = 'improved_model.pkl', encoder_path: str = 'improved_label_encoder.pkl'):
        """Initialize with improved model files."""
        self.line_extractor = ImprovedLineExtractor()
        self.model_data = None
        self.label_encoder = None
        
        # Try improved model first, fallback to original
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            self.load_model(model_path, encoder_path)
        else:
            print("Improved model not found, using original model...")
            self.load_model('model.pkl', 'label_encoder.pkl')
    
    def load_model(self, model_path: str, encoder_path: str):
        """Load the trained model and label encoder."""
        try:
            self.model_data = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            print(f"Model loaded from {model_path}")
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            raise
    
    def predict_line_labels(self, lines_data: List[Dict]) -> Tuple[List[str], List[float]]:
        """Predict labels with improved handling of missing features."""
        if not lines_data:
            return [], []
        
        # Prepare features
        df = pd.DataFrame(lines_data)
        feature_columns = self.model_data['feature_columns']
        
        # Handle missing columns by adding them with default values
        for col in feature_columns:
            if col not in df.columns:
                # Set reasonable defaults based on feature type
                if col.startswith('is_') or col.startswith('has_') or col.startswith('ends_'):
                    df[col] = False
                elif 'font_size' in col or 'position' in col:
                    df[col] = 0.0
                else:
                    df[col] = 0
        
        X = df[feature_columns].copy()
        
        # Convert boolean columns to numeric
        bool_columns = [col for col in feature_columns if col.startswith('is_') or col.startswith('has_') or col.startswith('ends_')]
        for col in bool_columns:
            if col in X.columns:
                X[col] = X[col].astype(int)
        
        # Handle NaN values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.model_data['scaler'].transform(X)
        
        # Predict
        predictions = self.model_data['model'].predict(X_scaled)
        probabilities = self.model_data['model'].predict_proba(X_scaled)
        
        # Get confidence scores
        confidence_scores = np.max(probabilities, axis=1)
        
        # Decode labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels.tolist(), confidence_scores.tolist()
    
    def extract_title_improved(self, lines_data: List[Dict], predicted_labels: List[str], 
                              confidence_scores: List[float]) -> str:
        """Improved title extraction with multiple strategies."""
        
        # Strategy 1: Use predicted title with high confidence
        title_candidates = []
        for i, (line, label, confidence) in enumerate(zip(lines_data, predicted_labels, confidence_scores)):
            if label == 'title' and confidence > 0.4:  # Lower threshold for title
                title_candidates.append({
                    'text': line['text'].strip(),
                    'confidence': confidence,
                    'page': line['page'],
                    'font_size': line['font_size_avg'],
                    'y_position': line['y_position']
                })
        
        if title_candidates:
            # Choose title with highest confidence, prefer first page
            first_page_titles = [t for t in title_candidates if t['page'] == 1]
            if first_page_titles:
                best_title = max(first_page_titles, key=lambda x: x['confidence'])
            else:
                best_title = max(title_candidates, key=lambda x: x['confidence'])
            return best_title['text']
        
        # Strategy 2: Look for large font on first page (top 3 lines)
        first_page_lines = [line for line in lines_data if line['page'] == 1]
        if first_page_lines:
            # Sort by y_position (top to bottom) and take top candidates
            first_page_lines.sort(key=lambda x: x['y_position'])
            top_candidates = first_page_lines[:5]  # Top 5 lines
            
            # Filter meaningful text (not too short, not just numbers)
            meaningful_candidates = []
            for line in top_candidates:
                text = line['text'].strip()
                if (len(text) > 10 and 
                    not re.match(r'^\d+$', text) and  # Not just numbers
                    not re.match(r'^page\s*\d+', text.lower()) and  # Not page numbers
                    not text.lower() in ['table of contents', 'contents', 'index']):
                    meaningful_candidates.append(line)
            
            if meaningful_candidates:
                # Choose the one with largest font size among top candidates
                best_candidate = max(meaningful_candidates, key=lambda x: x['font_size_avg'])
                return best_candidate['text'].strip()
        
        # Strategy 3: Look for text with distinctive formatting (large, bold, centered)
        formatted_candidates = []
        for line in lines_data:
            if (line['page'] <= 2 and  # First two pages
                line['font_size_avg'] > np.percentile([l['font_size_avg'] for l in lines_data], 75) and  # Large font
                len(line['text'].strip()) > 5):
                
                score = line['font_size_avg']
                if line.get('is_bold', False):
                    score += 2
                if line.get('is_centered', False):
                    score += 1
                if line['page'] == 1:
                    score += 3
                
                formatted_candidates.append({
                    'text': line['text'].strip(),
                    'score': score
                })
        
        if formatted_candidates:
            best_formatted = max(formatted_candidates, key=lambda x: x['score'])
            return best_formatted['text']
        
        return "Unknown Document"
    
    def extract_outline_improved(self, lines_data: List[Dict], predicted_labels: List[str], 
                                confidence_scores: List[float]) -> List[Dict]:
        """Improved outline extraction with adaptive thresholds."""
        outline = []
        heading_levels = ['H1', 'H2', 'H3']  # Removed H4
        seen_texts = set()
        
        # Calculate adaptive confidence threshold based on label distribution
        label_confidences = {}
        for label, confidence in zip(predicted_labels, confidence_scores):
            if label in heading_levels:
                if label not in label_confidences:
                    label_confidences[label] = []
                label_confidences[label].append(confidence)
        
        # Set adaptive thresholds per heading level
        adaptive_thresholds = {}
        for level in heading_levels:
            if level in label_confidences and label_confidences[level]:
                # Use median confidence as threshold, but not below 0.3
                threshold = max(0.3, np.percentile(label_confidences[level], 30))
                adaptive_thresholds[level] = threshold
            else:
                adaptive_thresholds[level] = 0.5
        
        print(f"Adaptive thresholds: {adaptive_thresholds}")
        
        for i, (line, label, confidence) in enumerate(zip(lines_data, predicted_labels, confidence_scores)):
            if label in heading_levels:
                threshold = adaptive_thresholds.get(label, 0.5)
                
                if confidence > threshold:
                    # Clean up text
                    text = line['text'].strip()
                    
                    # Skip very short headings or obvious noise
                    if len(text) < 2 or text.lower() in ['page', 'contents', 'index']:
                        continue
                    
                    # Normalize text for duplicate detection
                    normalized_text = re.sub(r'\s+', ' ', text.lower())
                    
                    # Skip duplicates
                    if normalized_text in seen_texts:
                        continue
                    
                    # Additional quality checks
                    if (len(text) >= 3 and 
                        not re.match(r'^\.*$', text) and  # Not just dots
                        not re.match(r'^\d+$', text)):    # Not just numbers
                        
                        outline_entry = {
                            'level': label,
                            'text': text,
                            'page': line['page']
                        }
                        outline.append(outline_entry)
                        seen_texts.add(normalized_text)
        
        # Sort by page number and then by y_position
        def sort_key(entry):
            # Find the original line data for y_position
            for line in lines_data:
                if (line['text'].strip() == entry['text'] and 
                    line['page'] == entry['page']):
                    return (entry['page'], line['y_position'])
            return (entry['page'], 0)
        
        outline.sort(key=sort_key)
        
        return outline
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """Process PDF with improved extraction."""
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
        
        # Extract title with improved method
        title = self.extract_title_improved(lines_data, predicted_labels, confidence_scores)
        
        # Extract outline with improved method
        outline = self.extract_outline_improved(lines_data, predicted_labels, confidence_scores)
        
        result = {
            'title': title,
            'outline': outline
        }
        
        print(f"Extracted title: {title}")
        print(f"Found {len(outline)} outline entries")
        
        return result
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all PDFs with improved extraction."""
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
    """Main function with improved processing."""
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # For local development
    if not os.path.exists(input_dir):
        input_dir = "input"
        output_dir = "output"
    
    try:
        generator = ImprovedOutlineGenerator()
        generator.process_directory(input_dir, output_dir)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
