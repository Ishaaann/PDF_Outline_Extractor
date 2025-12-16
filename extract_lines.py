"""
extract_lines.py - Line-level span extractor with feature builder
Extracts line-level features from PDF files for training and inference.
"""

import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
import re
import json


class LineExtractor:
    """Extracts line-level features from PDF documents."""
    
    def __init__(self):
        self.lines_data = []
    
    def extract_lines_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract line-level features from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing line features
        """
        doc = fitz.open(pdf_path)
        lines_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_rect = page.rect
            
            # Get text blocks with detailed formatting
            blocks = page.get_text("dict")
            
            # Group spans into lines
            lines = self._group_spans_into_lines(blocks, page_num + 1, page_rect)
            lines_data.extend(lines)
        
        doc.close()
        
        # Normalize features document-wide
        if lines_data:
            lines_data = self._normalize_features(lines_data)
        
        return lines_data
    
    def _group_spans_into_lines(self, blocks: Dict, page_num: int, page_rect) -> List[Dict]:
        """Group text spans into logical lines and extract features."""
        lines = []
        
        for block in blocks.get("blocks", []):
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                spans = line.get("spans", [])
                if not spans:
                    continue
                
                # Combine all spans in the line
                line_text = ""
                font_sizes = []
                is_bold = False
                is_italic = False
                x_coords = []
                y_coords = []
                
                for span in spans:
                    line_text += span.get("text", "")
                    font_sizes.append(span.get("size", 0))
                    
                    # Check for bold/italic in font name
                    font = span.get("font", "").lower()
                    if "bold" in font:
                        is_bold = True
                    if "italic" in font or "oblique" in font:
                        is_italic = True
                    
                    # Get coordinates
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    x_coords.append(bbox[0])
                    y_coords.append(bbox[1])
                
                # Skip empty lines
                line_text = line_text.strip()
                if not line_text:
                    continue
                
                # Calculate line features
                avg_font_size = np.mean(font_sizes) if font_sizes else 0
                max_font_size = max(font_sizes) if font_sizes else 0
                min_x = min(x_coords) if x_coords else 0
                min_y = min(y_coords) if y_coords else 0
                
                # Text analysis features
                word_count = len(line_text.split())
                char_length = len(line_text)
                is_title_case = line_text.istitle()
                is_upper_case = line_text.isupper()
                has_numbers = bool(re.search(r'\d', line_text))
                
                # Create line feature dictionary
                line_features = {
                    'text': line_text,
                    'page': page_num,
                    'font_size_avg': avg_font_size,
                    'font_size_max': max_font_size,
                    'is_bold': is_bold,
                    'is_italic': is_italic,
                    'x_position': min_x,
                    'y_position': min_y,
                    'word_count': word_count,
                    'char_length': char_length,
                    'is_title_case': is_title_case,
                    'is_upper_case': is_upper_case,
                    'has_numbers': has_numbers,
                    'page_width': page_rect.width,
                    'page_height': page_rect.height
                }
                
                lines.append(line_features)
        
        return lines
    
    def _normalize_features(self, lines_data: List[Dict]) -> List[Dict]:
        """Normalize features relative to the document."""
        if not lines_data:
            return lines_data
        
        # Extract numeric features for normalization
        font_sizes_avg = [line['font_size_avg'] for line in lines_data]
        font_sizes_max = [line['font_size_max'] for line in lines_data]
        x_positions = [line['x_position'] for line in lines_data]
        y_positions = [line['y_position'] for line in lines_data]
        
        # Calculate normalization parameters
        font_avg_mean = np.mean(font_sizes_avg)
        font_avg_std = np.std(font_sizes_avg) if np.std(font_sizes_avg) > 0 else 1
        
        font_max_mean = np.mean(font_sizes_max)
        font_max_std = np.std(font_sizes_max) if np.std(font_sizes_max) > 0 else 1
        
        # Normalize each line
        for line in lines_data:
            # Z-score normalization for font sizes
            line['font_size_avg_norm'] = (line['font_size_avg'] - font_avg_mean) / font_avg_std
            line['font_size_max_norm'] = (line['font_size_max'] - font_max_mean) / font_max_std
            
            # Relative position normalization (0-1 scale)
            line['x_position_norm'] = line['x_position'] / line['page_width'] if line['page_width'] > 0 else 0
            line['y_position_norm'] = line['y_position'] / line['page_height'] if line['page_height'] > 0 else 0
            
            # Relative font size compared to document max
            max_font_in_doc = max(font_sizes_max) if font_sizes_max else 1
            line['font_size_relative'] = line['font_size_avg'] / max_font_in_doc
        
        return lines_data
    
    def extract_and_save_features(self, input_dir: str, output_csv: str = "extracted_features.csv"):
        """
        Extract features from all PDFs in input directory and save to CSV.
        
        Args:
            input_dir: Directory containing PDF files
            output_csv: Output CSV file path
        """
        all_lines = []
        
        # Process all PDF files
        for filename in os.listdir(input_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_dir, filename)
                print(f"Processing {filename}...")
                
                try:
                    lines = self.extract_lines_from_pdf(pdf_path)
                    
                    # Add filename to each line
                    for line in lines:
                        line['filename'] = filename
                    
                    all_lines.extend(lines)
                    print(f"Extracted {len(lines)} lines from {filename}")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        # Save to CSV
        if all_lines:
            df = pd.DataFrame(all_lines)
            
            # Add empty label column for manual labeling
            df['label'] = 'None'  # Default label
            
            # Reorder columns to put label at the end for easier manual editing
            cols = [col for col in df.columns if col != 'label'] + ['label']
            df = df[cols]
            
            df.to_csv(output_csv, index=False)
            print(f"Saved {len(all_lines)} lines to {output_csv}")
            print("Please manually label the 'label' column with: title, H1, H2, H3, or None")
        else:
            print("No lines extracted from any PDF files.")


def main():
    """Main function to extract features from PDFs."""
    input_dir = "input"
    output_csv = "extracted_features.csv"
    
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' not found. Please create it and add PDF files.")
        return
    
    extractor = LineExtractor()
    extractor.extract_and_save_features(input_dir, output_csv)


if __name__ == "__main__":
    main()
