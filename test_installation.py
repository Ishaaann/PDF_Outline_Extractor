"""
test_installation.py - Test script to validate installation
Tests all components of the PDF outline extraction system.
"""

import sys
import importlib
import os
from pathlib import Path


def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing package imports...")
    
    required_packages = [
        'fitz',  # PyMuPDF
        'pandas',
        'numpy', 
        'sklearn',
        'joblib'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    print("✅ All packages imported successfully!")
    return True


def test_scripts():
    """Test if all Python scripts are valid."""
    print("\n🧪 Testing Python scripts...")
    
    scripts = [
        'extract_lines.py',
        'train_model.py', 
        'generate_outline.py',
        'run_pipeline.py'
    ]
    
    failed_scripts = []
    
    for script in scripts:
        try:
            with open(script, 'r') as f:
                compile(f.read(), script, 'exec')
            print(f"  ✅ {script}")
        except (FileNotFoundError, SyntaxError) as e:
            print(f"  ❌ {script}: {e}")
            failed_scripts.append(script)
    
    if failed_scripts:
        print(f"\n❌ Issues with scripts: {', '.join(failed_scripts)}")
        return False
    
    print("✅ All scripts are valid!")
    return True


def test_directories():
    """Test if required directories exist."""
    print("\n🧪 Testing directory structure...")
    
    required_dirs = ['input', 'output']
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✅ {dir_name}/ directory exists")
        else:
            print(f"  ⚠️  {dir_name}/ directory missing (will be created)")
            dir_path.mkdir(exist_ok=True)
            print(f"  ✅ Created {dir_name}/ directory")
    
    return True


def test_training_data():
    """Test if training data is available."""
    print("\n🧪 Testing training data...")
    
    if Path("label_lines.csv").exists():
        print("  ✅ Sample training data (label_lines.csv) found")
        
        try:
            import pandas as pd
            df = pd.read_csv("label_lines.csv")
            print(f"  ✅ Training data loaded: {len(df)} samples")
            
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                print(f"  📊 Label distribution: {dict(label_counts)}")
            else:
                print("  ⚠️  No 'label' column found in training data")
                
        except Exception as e:
            print(f"  ❌ Error reading training data: {e}")
            return False
    else:
        print("  ⚠️  No training data found (label_lines.csv)")
        print("  💡 Sample training data is provided for testing")
    
    return True


def test_model_files():
    """Test if model files exist."""
    print("\n🧪 Testing model files...")
    
    model_files = ['model.pkl', 'label_encoder.pkl']
    model_exists = all(Path(f).exists() for f in model_files)
    
    if model_exists:
        print("  ✅ Pre-trained model files found")
        try:
            import joblib
            model_data = joblib.load('model.pkl')
            label_encoder = joblib.load('label_encoder.pkl')
            print("  ✅ Model files loaded successfully")
        except Exception as e:
            print(f"  ❌ Error loading model files: {e}")
            return False
    else:
        print("  ⚠️  No pre-trained model found")
        print("  💡 Run train_model.py to create model files")
    
    return True


def create_sample_pdf_info():
    """Provide info about creating sample PDFs for testing."""
    print("\n📄 Sample PDF Testing:")
    print("  💡 To test the system, add PDF files to the input/ directory")
    print("  💡 The system works best with:")
    print("     - PDFs with clear heading structure")
    print("     - Text-based PDFs (not scanned images)")
    print("     - Documents ≤50 pages")
    print("  💡 Examples: Research papers, reports, manuals, books")


def main():
    """Run all tests."""
    print("🎯 Adobe Hackathon Round 1A - PDF Outline Extractor")
    print("🧪 Installation and Setup Test")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    tests = [
        test_imports,
        test_scripts,
        test_directories,
        test_training_data,
        test_model_files
    ]
    
    for test in tests:
        if not test():
            all_passed = False
    
    # Additional info
    create_sample_pdf_info()
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! System is ready to use.")
        print("🚀 Next steps:")
        print("   1. Add PDF files to input/ directory")
        print("   2. Run: python run_pipeline.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("💡 Check requirements.txt and README.md for setup instructions")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
