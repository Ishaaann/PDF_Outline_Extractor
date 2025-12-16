"""
improved_extract_lines.py - Enhanced line-level feature extractor
Improved version with additional features for better classification.
"""

import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
import re
import json


class ImprovedLineExtractor:
    """Enhanced line-level feature extractor with additional features."""
    
    def __init__(self):
        self.lines_data = []
    
    def extract_lines_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract enhanced line-level features from a PDF file.
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
        
        # Add contextual features
        if lines_data:
            lines_data = self._add_contextual_features(lines_data)
            lines_data = self._normalize_features(lines_data)
        
        return lines_data
    
    def _group_spans_into_lines(self, blocks: Dict, page_num: int, page_rect) -> List[Dict]:
        """Enhanced span grouping with better text analysis."""
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
                font_names = []
                is_bold = False
                is_italic = False
                x_coords = []
                y_coords = []
                colors = []
                
                for span in spans:
                    line_text += span.get("text", "")
                    font_sizes.append(span.get("size", 0))
                    font_names.append(span.get("font", ""))
                    
                    # Check for bold/italic in font name
                    font = span.get("font", "").lower()
                    if "bold" in font or "black" in font or "heavy" in font:
                        is_bold = True
                    if "italic" in font or "oblique" in font:
                        is_italic = True
                    
                    # Get coordinates
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    x_coords.append(bbox[0])
                    y_coords.append(bbox[1])
                    
                    # Color information
                    color = span.get("color", 0)
                    colors.append(color)
                
                # Skip empty lines
                line_text = line_text.strip()
                if not line_text:
                    continue
                
                # Calculate line features
                avg_font_size = np.mean(font_sizes) if font_sizes else 0
                max_font_size = max(font_sizes) if font_sizes else 0
                min_font_size = min(font_sizes) if font_sizes else 0
                font_size_variance = np.var(font_sizes) if len(font_sizes) > 1 else 0
                
                min_x = min(x_coords) if x_coords else 0
                max_x = max(x_coords) if x_coords else 0
                min_y = min(y_coords) if y_coords else 0
                line_width = max_x - min_x
                
                # Enhanced text analysis features
                word_count = len(line_text.split())
                char_length = len(line_text)
                is_title_case = line_text.istitle()
                is_upper_case = line_text.isupper()
                has_numbers = bool(re.search(r'\d', line_text))
                
                # New features
                starts_with_number = bool(re.match(r'^\d+\.?\s', line_text))
                is_centered = abs(min_x - (page_rect.width - max_x)) < 50  # Rough centering check
                has_colon = ':' in line_text
                ends_with_period = line_text.endswith('.')
                all_caps_words = len([w for w in line_text.split() if w.isupper() and len(w) > 1])
                
                # Punctuation analysis
                punctuation_count = len([c for c in line_text if c in '.,;:!?'])
                punctuation_ratio = punctuation_count / char_length if char_length > 0 else 0
                
                # Font consistency
                unique_fonts = len(set(font_names))
                font_consistency = 1.0 / unique_fonts if unique_fonts > 0 else 1.0
                
                # Indentation level (approximate)
                indentation_level = 0
                if min_x > 100:
                    indentation_level = 2
                elif min_x > 80:
                    indentation_level = 1
                
                # Create enhanced line feature dictionary
                line_features = {
                    'text': line_text,
                    'page': page_num,
                    'font_size_avg': avg_font_size,
                    'font_size_max': max_font_size,
                    'font_size_min': min_font_size,
                    'font_size_variance': font_size_variance,
                    'is_bold': is_bold,
                    'is_italic': is_italic,
                    'x_position': min_x,
                    'y_position': min_y,
                    'line_width': line_width,
                    'word_count': word_count,
                    'char_length': char_length,
                    'is_title_case': is_title_case,
                    'is_upper_case': is_upper_case,
                    'has_numbers': has_numbers,
                    'starts_with_number': starts_with_number,
                    'is_centered': is_centered,
                    'has_colon': has_colon,
                    'ends_with_period': ends_with_period,
                    'all_caps_words': all_caps_words,
                    'punctuation_ratio': punctuation_ratio,
                    'font_consistency': font_consistency,
                    'indentation_level': indentation_level,
                    'page_width': page_rect.width,
                    'page_height': page_rect.height
                }
                
                lines.append(line_features)
        
        return lines
    
    def _add_contextual_features(self, lines_data: List[Dict]) -> List[Dict]:
        """Add contextual features based on surrounding lines."""
        for i, line in enumerate(lines_data):
            # Previous line features
            if i > 0:
                prev_line = lines_data[i-1]
                line['prev_font_size'] = prev_line['font_size_avg']
                line['prev_is_bold'] = prev_line['is_bold']
                line['font_size_diff_prev'] = line['font_size_avg'] - prev_line['font_size_avg']
            else:
                line['prev_font_size'] = line['font_size_avg']
                line['prev_is_bold'] = line['is_bold']
                line['font_size_diff_prev'] = 0
            
            # Next line features
            if i < len(lines_data) - 1:
                next_line = lines_data[i+1]
                line['next_font_size'] = next_line['font_size_avg']
                line['next_is_bold'] = next_line['is_bold']
                line['font_size_diff_next'] = line['font_size_avg'] - next_line['font_size_avg']
            else:
                line['next_font_size'] = line['font_size_avg']
                line['next_is_bold'] = line['is_bold']
                line['font_size_diff_next'] = 0
            
            # Position in page
            page_lines = [l for l in lines_data if l['page'] == line['page']]
            line['position_in_page'] = len([l for l in page_lines if l['y_position'] < line['y_position']])
            line['lines_in_page'] = len(page_lines)
            line['relative_position_in_page'] = line['position_in_page'] / line['lines_in_page'] if line['lines_in_page'] > 0 else 0
        
        return lines_data
    
    def _normalize_features(self, lines_data: List[Dict]) -> List[Dict]:
        """Enhanced feature normalization."""
        if not lines_data:
            return lines_data
        
        # Extract numeric features for normalization
        font_sizes_avg = [line['font_size_avg'] for line in lines_data]
        font_sizes_max = [line['font_size_max'] for line in lines_data]
        x_positions = [line['x_position'] for line in lines_data]
        y_positions = [line['y_position'] for line in lines_data]
        line_widths = [line['line_width'] for line in lines_data]
        
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
            line['line_width_norm'] = line['line_width'] / line['page_width'] if line['page_width'] > 0 else 0
            
            # Relative font size compared to document max
            max_font_in_doc = max(font_sizes_max) if font_sizes_max else 1
            line['font_size_relative'] = line['font_size_avg'] / max_font_in_doc
            
            # Font size ranking (percentile)
            line['font_size_percentile'] = sum(1 for f in font_sizes_avg if f <= line['font_size_avg']) / len(font_sizes_avg)
        
        return lines_data
