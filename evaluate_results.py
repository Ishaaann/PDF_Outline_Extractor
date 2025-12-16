"""
evaluate_results.py - Compare generated results with ground truth
Evaluates the performance of the trained model against ground truth data.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import re


class ResultsEvaluator:
    """Evaluates generated outlines against ground truth."""
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra whitespace, numbers, and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'^\d+\.?\s*', '', text)  # Remove leading numbers
        return text.lower()
    
    def load_json(self, file_path: str) -> Dict:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def compare_titles(self, predicted_title: str, ground_truth_title: str) -> float:
        """Compare titles and return similarity score."""
        pred_norm = self.normalize_text(predicted_title)
        gt_norm = self.normalize_text(ground_truth_title)
        
        # Exact match
        if pred_norm == gt_norm:
            return 1.0
        
        # Partial match
        if pred_norm in gt_norm or gt_norm in pred_norm:
            shorter = min(len(pred_norm), len(gt_norm))
            longer = max(len(pred_norm), len(gt_norm))
            return shorter / longer if longer > 0 else 0.0
        
        return 0.0
    
    def compare_outlines(self, predicted_outline: List[Dict], ground_truth_outline: List[Dict]) -> Dict:
        """Compare outlines and return detailed metrics."""
        metrics = {
            'total_predicted': len(predicted_outline),
            'total_ground_truth': len(ground_truth_outline),
            'exact_matches': 0,
            'partial_matches': 0,
            'level_matches': 0,
            'text_matches': 0,
            'page_matches': 0
        }
        
        # Create normalized versions of ground truth for matching
        gt_normalized = []
        for item in ground_truth_outline:
            gt_normalized.append({
                'text_norm': self.normalize_text(item['text']),
                'level': item['level'],
                'page': item['page'],
                'original': item
            })
        
        # Compare each predicted item
        for pred_item in predicted_outline:
            pred_text_norm = self.normalize_text(pred_item['text'])
            
            best_match = None
            best_score = 0
            
            for gt_item in gt_normalized:
                # Text similarity
                if pred_text_norm == gt_item['text_norm']:
                    text_score = 1.0
                elif pred_text_norm in gt_item['text_norm'] or gt_item['text_norm'] in pred_text_norm:
                    shorter = min(len(pred_text_norm), len(gt_item['text_norm']))
                    longer = max(len(pred_text_norm), len(gt_item['text_norm']))
                    text_score = shorter / longer if longer > 0 else 0.0
                else:
                    text_score = 0.0
                
                if text_score > best_score:
                    best_score = text_score
                    best_match = gt_item
            
            # Count matches based on best match found
            if best_match and best_score > 0.7:  # Threshold for considering a match
                if best_score == 1.0 and pred_item['level'] == best_match['level']:
                    metrics['exact_matches'] += 1
                else:
                    metrics['partial_matches'] += 1
                
                if pred_item['level'] == best_match['level']:
                    metrics['level_matches'] += 1
                
                if best_score > 0.8:
                    metrics['text_matches'] += 1
                
                if abs(pred_item['page'] - best_match['page']) <= 1:  # Allow 1 page difference
                    metrics['page_matches'] += 1
        
        return metrics
    
    def evaluate_file(self, predicted_path: str, ground_truth_path: str) -> Dict:
        """Evaluate a single file."""
        try:
            predicted = self.load_json(predicted_path)
            ground_truth = self.load_json(ground_truth_path)
            
            # Compare titles
            title_score = self.compare_titles(predicted['title'], ground_truth['title'])
            
            # Compare outlines
            outline_metrics = self.compare_outlines(predicted['outline'], ground_truth['outline'])
            
            return {
                'title_score': title_score,
                'outline_metrics': outline_metrics,
                'predicted_title': predicted['title'],
                'ground_truth_title': ground_truth['title']
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def evaluate_all(self, output_dir: str = 'output', ground_truth_dir: str = 'data'):
        """Evaluate all files and generate summary report."""
        output_path = Path(output_dir)
        gt_path = Path(ground_truth_dir)
        
        if not output_path.exists() or not gt_path.exists():
            print(f"Directories not found: {output_dir} or {ground_truth_dir}")
            return
        
        results = {}
        overall_metrics = {
            'total_files': 0,
            'title_scores': [],
            'total_predicted_headings': 0,
            'total_ground_truth_headings': 0,
            'total_exact_matches': 0,
            'total_partial_matches': 0,
            'total_level_matches': 0,
            'total_text_matches': 0,
            'total_page_matches': 0
        }
        
        # Process each file
        for pred_file in output_path.glob("*.json"):
            gt_file = gt_path / pred_file.name
            
            if gt_file.exists():
                result = self.evaluate_file(str(pred_file), str(gt_file))
                results[pred_file.name] = result
                
                if 'error' not in result:
                    overall_metrics['total_files'] += 1
                    overall_metrics['title_scores'].append(result['title_score'])
                    
                    outline_metrics = result['outline_metrics']
                    overall_metrics['total_predicted_headings'] += outline_metrics['total_predicted']
                    overall_metrics['total_ground_truth_headings'] += outline_metrics['total_ground_truth']
                    overall_metrics['total_exact_matches'] += outline_metrics['exact_matches']
                    overall_metrics['total_partial_matches'] += outline_metrics['partial_matches']
                    overall_metrics['total_level_matches'] += outline_metrics['level_matches']
                    overall_metrics['total_text_matches'] += outline_metrics['text_matches']
                    overall_metrics['total_page_matches'] += outline_metrics['page_matches']
        
        # Generate report
        self.print_evaluation_report(results, overall_metrics)
    
    def print_evaluation_report(self, results: Dict, overall_metrics: Dict):
        """Print detailed evaluation report."""
        print("="*80)
        print("📊 PDF Outline Extraction - Evaluation Report")
        print("="*80)
        
        print(f"\n📁 Files evaluated: {overall_metrics['total_files']}")
        
        # Title evaluation
        if overall_metrics['title_scores']:
            avg_title_score = sum(overall_metrics['title_scores']) / len(overall_metrics['title_scores'])
            print(f"\n📄 Title Extraction:")
            print(f"   Average similarity score: {avg_title_score:.3f}")
            perfect_titles = sum(1 for score in overall_metrics['title_scores'] if score == 1.0)
            print(f"   Perfect matches: {perfect_titles}/{len(overall_metrics['title_scores'])} ({perfect_titles/len(overall_metrics['title_scores'])*100:.1f}%)")
        
        # Outline evaluation
        print(f"\n📋 Outline Extraction:")
        print(f"   Predicted headings: {overall_metrics['total_predicted_headings']}")
        print(f"   Ground truth headings: {overall_metrics['total_ground_truth_headings']}")
        
        if overall_metrics['total_ground_truth_headings'] > 0:
            recall = overall_metrics['total_exact_matches'] / overall_metrics['total_ground_truth_headings']
            print(f"   Exact matches: {overall_metrics['total_exact_matches']} (Recall: {recall:.3f})")
            
            level_accuracy = overall_metrics['total_level_matches'] / overall_metrics['total_predicted_headings'] if overall_metrics['total_predicted_headings'] > 0 else 0
            print(f"   Level accuracy: {level_accuracy:.3f}")
            
            text_accuracy = overall_metrics['total_text_matches'] / overall_metrics['total_predicted_headings'] if overall_metrics['total_predicted_headings'] > 0 else 0
            print(f"   Text accuracy: {text_accuracy:.3f}")
        
        # Detailed file results
        print(f"\n📁 File-by-file results:")
        for filename, result in results.items():
            if 'error' in result:
                print(f"   ❌ {filename}: {result['error']}")
            else:
                title_score = result['title_score']
                outline_metrics = result['outline_metrics']
                
                exact_matches = outline_metrics['exact_matches']
                total_gt = outline_metrics['total_ground_truth']
                
                print(f"   📄 {filename}:")
                print(f"      Title: {title_score:.2f} | Headings: {exact_matches}/{total_gt}")
                print(f"      Predicted: '{result['predicted_title'][:50]}...'")
                print(f"      Expected:  '{result['ground_truth_title'][:50]}...'")
        
        print("="*80)


def main():
    """Main evaluation function."""
    evaluator = ResultsEvaluator()
    evaluator.evaluate_all()


if __name__ == "__main__":
    main()
