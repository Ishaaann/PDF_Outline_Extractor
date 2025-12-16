"""
create_training_data.py - Generate training data from PDF-JSON pairs
Extracts features from PDFs and creates labeled training data using corresponding JSON files.
"""

import os
import json
import pandas as pd
from pathlib import Path
from extract_lines import LineExtractor
from typing import Dict, List, Tuple
import re


class TrainingDataGenerator:
    """Generates labeled training data from PDF-JSON pairs."""
    
    def __init__(self):
        self.line_extractor = LineExtractor()
    
    def load_ground_truth(self, json_path: str) -> Dict:
        """Load ground truth data from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        return text.lower()
    
    def find_text_matches(self, line_text: str, outline_texts: List[str]) -> str:
        """Find the best matching outline text for a line."""
        line_normalized = self.normalize_text(line_text)
        
        best_match = "None"
        best_score = 0
        
        for outline_item in outline_texts:
            outline_normalized = self.normalize_text(outline_item['text'])
            
            # Exact match
            if line_normalized == outline_normalized:
                level = outline_item['level']
                # Remap H4 to H3 since we don't want H4 classification
                if level == 'H4':
                    level = 'H3'
                return level
            
            # Partial match - check if one contains the other
            if outline_normalized in line_normalized or line_normalized in outline_normalized:
                # Calculate similarity score
                shorter = min(len(line_normalized), len(outline_normalized))
                longer = max(len(line_normalized), len(outline_normalized))
                score = shorter / longer if longer > 0 else 0
                
                if score > best_score and score > 0.7:  # Threshold for partial match
                    level = outline_item['level']
                    # Remap H4 to H3 since we don't want H4 classification
                    if level == 'H4':
                        level = 'H3'
                    best_match = level
                    best_score = score
        
        return best_match
    
    def find_title_match(self, line_text: str, title: str) -> bool:
        """Check if a line matches the document title."""
        line_normalized = self.normalize_text(line_text)
        title_normalized = self.normalize_text(title)
        
        # Exact match
        if line_normalized == title_normalized:
            return True
        
        # Partial match for title
        if title_normalized in line_normalized or line_normalized in title_normalized:
            shorter = min(len(line_normalized), len(title_normalized))
            longer = max(len(line_normalized), len(title_normalized))
            score = shorter / longer if longer > 0 else 0
            return score > 0.6  # Lower threshold for title matching
        
        return False
    
    def label_lines(self, lines_data: List[Dict], ground_truth: Dict) -> List[Dict]:
        """Label extracted lines based on ground truth data."""
        title = ground_truth.get('title', '')
        outline = ground_truth.get('outline', [])
        
        # Create a list of all outline items with their levels
        outline_items = []
        for item in outline:
            outline_items.append({
                'text': item['text'],
                'level': item['level'],
                'page': item['page']
            })
        
        labeled_lines = []
        
        for line in lines_data:
            line_copy = line.copy()
            
            # Check if this line is the title
            if self.find_title_match(line['text'], title):
                line_copy['label'] = 'title'
            else:
                # Check if this line matches any outline item
                label = self.find_text_matches(line['text'], outline_items)
                line_copy['label'] = label
            
            labeled_lines.append(line_copy)
        
        return labeled_lines
    
    def process_pdf_json_pair(self, pdf_path: str, json_path: str) -> List[Dict]:
        """Process a single PDF-JSON pair and return labeled data."""
        print(f"Processing {os.path.basename(pdf_path)}...")
        
        # Extract features from PDF
        lines_data = self.line_extractor.extract_lines_from_pdf(pdf_path)
        
        # Load ground truth
        ground_truth = self.load_ground_truth(json_path)
        
        # Label the lines
        labeled_lines = self.label_lines(lines_data, ground_truth)
        
        # Add filename for reference
        for line in labeled_lines:
            line['filename'] = os.path.basename(pdf_path)
        
        print(f"  Extracted {len(labeled_lines)} lines")
        
        # Count labels
        label_counts = {}
        for line in labeled_lines:
            label = line['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"  Label distribution: {label_counts}")
        
        return labeled_lines
    
    def generate_training_data(self, input_dir: str, data_dir: str, output_csv: str = "training_data.csv"):
        """Generate training data from all PDF-JSON pairs."""
        input_path = Path(input_dir)
        data_path = Path(data_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        all_labeled_data = []
        
        # Find all PDF files
        pdf_files = list(input_path.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            # Find corresponding JSON file
            json_file = data_path / (pdf_file.stem + ".json")
            
            if json_file.exists():
                try:
                    labeled_data = self.process_pdf_json_pair(str(pdf_file), str(json_file))
                    all_labeled_data.extend(labeled_data)
                except Exception as e:
                    print(f"Error processing {pdf_file.name}: {e}")
                    continue
            else:
                print(f"Warning: No corresponding JSON found for {pdf_file.name}")
        
        if all_labeled_data:
            # Convert to DataFrame and save
            df = pd.DataFrame(all_labeled_data)
            
            # Reorder columns to put label at the end
            cols = [col for col in df.columns if col != 'label'] + ['label']
            df = df[cols]
            
            df.to_csv(output_csv, index=False)
            print(f"\nTraining data saved to {output_csv}")
            print(f"Total samples: {len(all_labeled_data)}")
            
            # Show overall label distribution
            label_counts = df['label'].value_counts()
            print(f"Overall label distribution:")
            for label, count in label_counts.items():
                print(f"  {label}: {count}")
            
            return output_csv
        else:
            print("No training data generated!")
            return None


def main():
    """Main function to generate training data."""
    input_dir = "input"
    data_dir = "data"
    output_csv = "training_data.csv"
    
    generator = TrainingDataGenerator()
    
    try:
        result = generator.generate_training_data(input_dir, data_dir, output_csv)
        if result:
            print(f"\n✅ Training data generation completed!")
            print(f"📄 Generated file: {result}")
            print(f"\n🚀 Next step: Run 'python train_model.py' to train the model")
        else:
            print("❌ Failed to generate training data")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
