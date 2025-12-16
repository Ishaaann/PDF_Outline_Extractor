# Adobe Hackathon Round 1A - PDF Outline Extractor

A Python-based ML-driven system that extracts structured outlines from PDF documents using line-level analysis and relative feature normalization.

## 🎯 Overview

This project processes PDF files (≤50 pages) and outputs structured JSON files containing:
- Document title
- Hierarchical outline (H1, H2, H3 headings) with page numbers

## 🧠 Key Features

- **Line-Based Text Extraction**: Uses PyMuPDF for precise line-level text extraction
- **Relative Feature Analysis**: Normalizes font sizes and positions per document
- **ML-Driven Classification**: Trained supervised model for heading detection
- **Offline Processing**: CPU-only, no external dependencies
- **Docker Ready**: Containerized for easy deployment

## 📁 Project Structure

```
adobe_round1_A/
├── extract_lines.py         # Line-level feature extractor
├── label_lines.csv          # Manually labeled training data
├── train_model.py           # ML model training script
├── generate_outline.py      # Main inference script
├── model.pkl               # Trained model (generated)
├── label_encoder.pkl       # Label encoder (generated)
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── input/                 # PDF input directory
└── output/               # JSON output directory
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Or using conda
conda create -n pdf_extractor python=3.10
conda activate pdf_extractor
pip install -r requirements.txt
```

### 2. Train the Model (First Time Setup)

```bash
# Extract features from sample PDFs (place PDFs in input/ directory)
python extract_lines.py

# Manually label the generated CSV file
# Edit label_lines.csv and set appropriate labels: title, H1, H2, H3, None

# Train the model
python train_model.py
```

### 3. Process PDFs

```bash
# Place PDF files in input/ directory
cp your_pdfs/*.pdf input/

# Generate outlines
python generate_outline.py

# Check results in output/ directory
ls output/*.json
```

### 4. Docker Deployment

```bash
# Build container
docker build -t pdf-outline-extractor .

# Run with volume mounts
docker run -v /path/to/pdfs:/app/input -v /path/to/output:/app/output pdf-outline-extractor
```

## 📊 Output Format

Each PDF generates a corresponding JSON file:

```json
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Chapter 1", "page": 1 },
    { "level": "H2", "text": "Overview", "page": 2 },
    { "level": "H3", "text": "Details", "page": 2 }
  ]
}
```

## 🔧 Training Your Own Model

### 1. Extract Features

```python
from extract_lines import LineExtractor

extractor = LineExtractor()
extractor.extract_and_save_features("input", "my_features.csv")
```

### 2. Manual Labeling

Edit the generated CSV file and label the `label` column with:
- `title`: Document title
- `H1`: Main headings
- `H2`: Sub-headings  
- `H3`: Sub-sub-headings
- `None`: Regular text

### 3. Train Model

```python
from train_model import OutlineModelTrainer

trainer = OutlineModelTrainer()
X, y = trainer.load_training_data("my_features.csv")
trainer.train_model(X, y, model_type='random_forest')
trainer.save_model('model.pkl', 'label_encoder.pkl')
```

## 🛠️ Technical Details

### Feature Engineering

The system extracts these line-level features:
- **Font Features**: Average/max font size (normalized)
- **Formatting**: Bold, italic flags
- **Position**: X/Y coordinates (normalized to page dimensions)
- **Text Analysis**: Word count, character length, title case, uppercase
- **Relative Metrics**: Font size relative to document maximum

### ML Model

- **Algorithm**: Random Forest Classifier (default)
- **Features**: 12 engineered features per line
- **Normalization**: Z-score for font sizes, 0-1 scale for positions
- **Output**: 5-class classification (title, H1, H2, H3, None)

### Performance

- **Speed**: <10 seconds for 50-page documents
- **Model Size**: <200MB total
- **Accuracy**: Typically >85% on well-formatted documents

## 🐛 Troubleshooting

### Common Issues

1. **No PDFs found**: Ensure PDF files are in the `input/` directory
2. **Model not found**: Run `train_model.py` first to generate model files
3. **Poor extraction**: Check PDF quality and ensure text is not embedded as images
4. **Low accuracy**: Add more labeled training data for your document types

### Debug Mode

```python
# Enable verbose output
import logging
logging.basicConfig(level=logging.DEBUG)

# Check extracted features
from extract_lines import LineExtractor
extractor = LineExtractor()
lines = extractor.extract_lines_from_pdf("input/sample.pdf")
print(f"Extracted {len(lines)} lines")
```

## 📝 Development Notes

### Extending the System

1. **Add New Features**: Modify `extract_lines.py` to add more text analysis features
2. **Try Different Models**: Edit `train_model.py` to experiment with other algorithms
3. **Custom Post-processing**: Modify `generate_outline.py` for domain-specific rules

### Model Improvements

- Add more training data for better accuracy
- Experiment with ensemble methods
- Include context features (surrounding lines)
- Add confidence thresholding for outline filtering

## 📄 License

This project is created for the Adobe Hackathon Round 1A competition.

## 🤝 Contributing

1. Add more diverse training data
2. Improve feature engineering
3. Optimize model performance
4. Add support for additional heading levels
5. Enhance error handling and validation
