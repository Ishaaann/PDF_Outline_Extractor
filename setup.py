#!/usr/bin/env python3
"""
setup.py - Setup script for Adobe Hackathon PDF Outline Extractor
Initializes the project and trains the model with sample data.
"""

import os
import sys
import subprocess
from pathlib import Path


def install_requirements():
    """Install required Python packages."""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("📁 Creating project directories...")
    
    dirs = ["input", "output"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created {dir_name}/ directory")


def train_initial_model():
    """Train the initial model with sample data."""
    print("🧠 Training initial model with sample data...")
    
    if Path("model.pkl").exists() and Path("label_encoder.pkl").exists():
        print("✅ Model files already exist, skipping training")
        return True
    
    try:
        # Import and run training
        from train_model import OutlineModelTrainer
        
        trainer = OutlineModelTrainer()
        X, y = trainer.load_training_data("label_lines.csv")
        trainer.train_model(X, y, model_type='random_forest')
        trainer.save_model('model.pkl', 'label_encoder.pkl')
        
        print("✅ Model trained successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False


def test_system():
    """Test the complete system."""
    print("🧪 Testing system components...")
    
    try:
        # Test imports
        import fitz
        import pandas as pd
        import numpy as np
        import sklearn
        import joblib
        
        print("✅ All imports successful")
        
        # Test model loading
        if Path("model.pkl").exists():
            model_data = joblib.load("model.pkl")
            label_encoder = joblib.load("label_encoder.pkl")
            print("✅ Model files loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False


def show_usage_instructions():
    """Show instructions for using the system."""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Add PDF files to the 'input/' directory")
    print("2. Run the extraction pipeline:")
    print("   python run_pipeline.py")
    print("\n💡 Alternative commands:")
    print("   python generate_outline.py    # Direct outline generation")
    print("   python test_installation.py   # Test installation")
    print("\n🐳 Docker Usage:")
    print("   docker-compose up             # Run with Docker")
    print("   docker-compose --profile training up pdf-trainer  # Train new model")
    print("\n📊 Expected Output:")
    print("   - JSON files in 'output/' directory")
    print("   - Each PDF gets a corresponding .json outline")
    print("\n📄 JSON Format:")
    print('   {')
    print('     "title": "Document Title",')
    print('     "outline": [')
    print('       {"level": "H1", "text": "Chapter 1", "page": 1},')
    print('       {"level": "H2", "text": "Overview", "page": 2}')
    print('     ]')
    print('   }')
    print("\n" + "="*60)


def main():
    """Main setup function."""
    print("🎯 Adobe Hackathon Round 1A - PDF Outline Extractor Setup")
    print("🚀 Initializing project...")
    print("="*60)
    
    success = True
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Install requirements
    if not install_requirements():
        success = False
    
    # Step 3: Train initial model
    if success and not train_initial_model():
        success = False
    
    # Step 4: Test system
    if success and not test_system():
        success = False
    
    # Step 5: Show instructions
    if success:
        show_usage_instructions()
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        print("💡 Try running: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
