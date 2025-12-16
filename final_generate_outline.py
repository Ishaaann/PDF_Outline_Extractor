"""
final_generate_outline.py - Final optimized inference script
Optimized version that balances precision and recall.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from improved_extract_lines import ImprovedLineExtractor
import re


class FinalOutlineGenerator:
    """Final optimized outline generator."""
    
    def __init__(self):
        self.line_extractor = ImprovedLineExtractor()
        self.model_data = None
        self.label_encoder = None
        
        # Load best available model
        if os.path.exists('improved_model.pkl'):
            self.load_model('improved_model.pkl', 'improved_label_encoder.pkl')
        else:
            self.load_model('model.pkl', 'label_encoder.pkl')
    
    def load_model(self, model_path: str, encoder_path: str):
        """Load model files."""
        self.model_data = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        print(f"Model loaded from {model_path}")
    
    def predict_line_labels(self, lines_data: List[Dict]) -> Tuple[List[str], List[float]]:
        """Predict with robust feature handling."""
        if not lines_data:
            return [], []
        
        df = pd.DataFrame(lines_data)
        feature_columns = self.model_data['feature_columns']
        
        # Handle missing features
        for col in feature_columns:
            if col not in df.columns:
                if col.startswith('is_') or col.startswith('has_') or col.startswith('ends_'):
                    df[col] = False
                else:
                    df[col] = 0.0
        
        X = df[feature_columns].copy()
        
        # Convert boolean columns
        bool_columns = [col for col in feature_columns if col.startswith('is_') or col.startswith('has_') or col.startswith('ends_')]
        for col in bool_columns:
            if col in X.columns:
                X[col] = X[col].astype(int)
        
        X = X.fillna(0)
        X_scaled = self.model_data['scaler'].transform(X)
        
        predictions = self.model_data['model'].predict(X_scaled)
        probabilities = self.model_data['model'].predict_proba(X_scaled)
        confidence_scores = np.max(probabilities, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels.tolist(), confidence_scores.tolist()
    
    def extract_title_final(self, lines_data: List[Dict], predicted_labels: List[str], 
                           confidence_scores: List[float]) -> str:
        """Final title extraction with multiple fallback strategies."""
        
        # Strategy 1: Predicted titles
        title_candidates = []
        for i, (line, label, confidence) in enumerate(zip(lines_data, predicted_labels, confidence_scores)):
            if label == 'title' and confidence > 0.3:
                title_candidates.append({
                    'text': line['text'].strip(),
                    'confidence': confidence,
                    'page': line['page'],
                    'y_position': line['y_position']
                })
        
        if title_candidates:
            # Prefer first page, then highest confidence
            first_page_titles = [t for t in title_candidates if t['page'] == 1]
            if first_page_titles:
                best_title = max(first_page_titles, key=lambda x: x['confidence'])
            else:
                best_title = max(title_candidates, key=lambda x: x['confidence'])
            return best_title['text']
        
        # Strategy 2: First page analysis with multiple criteria
        first_page_lines = [line for line in lines_data if line['page'] == 1]
        if first_page_lines:
            first_page_lines.sort(key=lambda x: x['y_position'])
            
            # Score each line based on multiple factors
            candidates = []
            for i, line in enumerate(first_page_lines[:10]):  # Top 10 lines
                text = line['text'].strip()
                
                # Skip obvious non-titles
                if (len(text) < 5 or 
                    re.match(r'^\d+$', text) or 
                    text.lower() in ['page', 'contents', 'table of contents', 'index'] or
                    re.match(r'^page\s*\d+', text.lower())):
                    continue
                
                score = 0
                
                # Font size score (larger is better)
                font_percentile = line.get('font_size_percentile', 0)
                score += font_percentile * 40
                
                # Position score (earlier is better, but not first line which might be header)
                if i == 0:
                    score += 10  # First line bonus, but not too much
                elif i <= 3:
                    score += 20  # Lines 2-4 are often titles
                
                # Formatting scores
                if line.get('is_bold', False):
                    score += 15
                if line.get('is_centered', False):
                    score += 10
                if line.get('is_title_case', False):
                    score += 5
                
                # Length score (reasonable length)
                if 10 <= len(text) <= 100:
                    score += 10
                elif len(text) > 100:
                    score -= 5
                
                # Avoid lines with too much punctuation (likely body text)
                punct_ratio = line.get('punctuation_ratio', 0)
                if punct_ratio > 0.1:
                    score -= 20
                
                candidates.append({
                    'text': text,
                    'score': score,
                    'line': line
                })
            
            if candidates:
                best_candidate = max(candidates, key=lambda x: x['score'])
                return best_candidate['text']
        
        return "Unknown Document"
    
    def extract_outline_final(self, lines_data: List[Dict], predicted_labels: List[str], 
                             confidence_scores: List[float]) -> List[Dict]:
        """Final outline extraction with optimized thresholds."""
        outline = []
        heading_levels = ['H1', 'H2', 'H3']  # Removed H4
        seen_texts = set()
        
        # Use lower base thresholds to improve recall
        base_thresholds = {'H1': 0.4, 'H2': 0.45, 'H3': 0.5}  # Removed H4
        
        # Collect all heading predictions
        heading_predictions = []
        for i, (line, label, confidence) in enumerate(zip(lines_data, predicted_labels, confidence_scores)):
            if label in heading_levels:
                heading_predictions.append({
                    'line': line,
                    'label': label,
                    'confidence': confidence,
                    'index': i
                })
        
        # Sort by confidence for each level
        by_level = {}
        for pred in heading_predictions:
            level = pred['label']
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(pred)
        
        # Apply adaptive thresholds
        for level in heading_levels:
            if level in by_level:
                predictions = by_level[level]
                predictions.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Use base threshold but adapt based on distribution
                threshold = base_thresholds[level]
                
                # If we have many predictions, be more selective
                if len(predictions) > 20:
                    threshold = max(threshold, np.percentile([p['confidence'] for p in predictions], 70))
                
                for pred in predictions:
                    if pred['confidence'] >= threshold:
                        line = pred['line']
                        text = line['text'].strip()
                        
                        # Quality checks
                        if (len(text) >= 3 and 
                            not re.match(r'^\.*$', text) and
                            not re.match(r'^\d+$', text) and
                            text.lower() not in ['page', 'contents', 'index']):
                            
                            # Improved normalization for duplicate detection
                            normalized_text = re.sub(r'\s+', ' ', text.lower().strip())
                            # Remove common prefixes like numbers for better duplicate detection
                            normalized_for_dup = re.sub(r'^\d+\.?\d*\.?\s*', '', normalized_text)
                            
                            # Check for duplicates using both full text and without numbering
                            is_duplicate = (normalized_text in seen_texts or 
                                          normalized_for_dup in seen_texts or
                                          any(normalized_for_dup in existing for existing in seen_texts) or
                                          any(existing in normalized_for_dup for existing in seen_texts if len(existing) > 5))
                            
                            if not is_duplicate:
                                outline.append({
                                    'level': pred['label'],
                                    'text': text,
                                    'page': line['page']
                                })
                                seen_texts.add(normalized_text)
                                seen_texts.add(normalized_for_dup)        # Sort by page and position
        def sort_key(entry):
            for line in lines_data:
                if (line['text'].strip() == entry['text'] and 
                    line['page'] == entry['page']):
                    return (entry['page'], line['y_position'])
            return (entry['page'], 0)
        
        outline.sort(key=sort_key)
        return outline
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """Process PDF with final optimized extraction."""
        print(f"Processing {os.path.basename(pdf_path)}...")
        
        lines_data = self.line_extractor.extract_lines_from_pdf(pdf_path)
        
        if not lines_data:
            return {'title': 'Empty Document', 'outline': []}
        
        predicted_labels, confidence_scores = self.predict_line_labels(lines_data)
        title = self.extract_title_final(lines_data, predicted_labels, confidence_scores)
        outline = self.extract_outline_final(lines_data, predicted_labels, confidence_scores)
        
        result = {'title': title, 'outline': outline}
        
        print(f"Extracted title: {title}")
        print(f"Found {len(outline)} outline entries")
        
        return result
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all PDFs in directory."""
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
                result = self.process_pdf(pdf_path)
                
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
    """Main function."""
    input_dir = "/app/input" if os.path.exists("/app/input") else "input"
    output_dir = "/app/output" if os.path.exists("/app/output") else "output"
    
    try:
        generator = FinalOutlineGenerator()
        generator.process_directory(input_dir, output_dir)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
