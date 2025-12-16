"""
run_pipeline.py - Complete pipeline runner
Runs the complete PDF outline extraction pipeline.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def check_prerequisites():
    """Check if required files and directories exist."""
    print("🔍 Checking prerequisites...")
    
    # Check if input directory exists and has PDFs
    input_dir = Path("input")
    if not input_dir.exists():
        print("❌ Input directory not found. Creating it...")
        input_dir.mkdir()
        print("📁 Please add PDF files to the 'input' directory and run again.")
        return False
    
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in input directory.")
        print("📁 Please add PDF files to the 'input' directory and run again.")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF files in input directory")
    
    # Check if output directory exists
    output_dir = Path("output")
    if not output_dir.exists():
        output_dir.mkdir()
        print("✅ Created output directory")
    
    return True


def main():
    """Main pipeline runner."""
    print("🎯 Adobe Hackathon Round 1A - PDF Outline Extractor Pipeline")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Step 1: Check if improved model exists, if not, train it
    if not (Path("improved_model.pkl").exists() and Path("improved_label_encoder.pkl").exists()):
        print("\n🔧 No improved model found. Training enhanced model...")
        
        # Train improved model
        if not run_command("python improved_train_model.py", "Training improved ML model with enhanced features"):
            print("❌ Improved model training failed! Falling back to basic model...")
            
            # Fallback to basic training if improved fails
            if not (Path("model.pkl").exists() and Path("label_encoder.pkl").exists()):
                if not run_command("python train_model.py", "Training basic ML model"):
                    print("❌ Basic model training also failed!")
                    sys.exit(1)
    else:
        print("✅ Found existing improved model")
    
    # Step 2: Generate outlines using the best available model
    if not run_command("python final_generate_outline.py", "Generating PDF outlines with optimized inference"):
        print("❌ Outline generation failed!")
        sys.exit(1)
    
    # Step 3: Show results
    output_dir = Path("output")
    json_files = list(output_dir.glob("*.json"))
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📊 Generated {len(json_files)} outline files:")
    
    for json_file in json_files:
        print(f"  📄 {json_file.name}")
    
    print(f"\n📁 Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
