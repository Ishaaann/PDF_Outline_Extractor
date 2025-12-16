"""
test_generate_outline.py - Run the model on test documents
"""

import os
import json
from final_generate_outline import FinalOutlineGenerator


def process_test_documents():
    """Process documents in the test folder and create outputs."""
    
    # Setup paths
    test_input_dir = "test"
    test_output_dir = "test_output"
    
    # Create output directory
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Initialize the generator
    generator = FinalOutlineGenerator()
    
    # Check if test directory exists
    if not os.path.exists(test_input_dir):
        print(f"Test directory '{test_input_dir}' not found.")
        return
    
    # Find PDF files in test directory
    pdf_files = [f for f in os.listdir(test_input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{test_input_dir}'")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) in test directory")
    print("=" * 60)
    
    for pdf_file in pdf_files:
        try:
            pdf_path = os.path.join(test_input_dir, pdf_file)
            print(f"Processing test file: {pdf_file}")
            
            # Process the PDF
            result = generator.process_pdf(pdf_path)
            
            # Create output filename
            output_filename = os.path.splitext(pdf_file)[0] + '.json'
            output_path = os.path.join(test_output_dir, output_filename)
            
            # Save the result
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved outline to {output_path}")
            print(f"   Title: {result['title']}")
            print(f"   Outline entries: {len(result['outline'])}")
            
            # Show outline structure
            if result['outline']:
                print("   Outline structure:")
                for i, entry in enumerate(result['outline'][:10]):  # Show first 10 entries
                    print(f"     {entry['level']}: {entry['text'][:50]}{'...' if len(entry['text']) > 50 else ''} (page {entry['page']})")
                if len(result['outline']) > 10:
                    print(f"     ... and {len(result['outline']) - 10} more entries")
            
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")
            continue
    
    print(f"✅ Test processing completed. Results saved to '{test_output_dir}'")


if __name__ == "__main__":
    process_test_documents()
