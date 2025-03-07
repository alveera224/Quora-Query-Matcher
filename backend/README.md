# 🚀 Quora Query Matcher Backend

![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange)
![BERT](https://img.shields.io/badge/BERT-Transformers-blue)
![GPU](https://img.shields.io/badge/GPU-Accelerated-red)

> The powerful AI engine behind Quora Query Matcher, leveraging state-of-the-art BERT models to detect semantically similar questions with high accuracy.

## ✨ Features

- **High-performance API** built with FastAPI for async operations
- **Advanced BERT model** fine-tuned on 90,000+ Quora question pairs
- **GPU acceleration** for lightning-fast inference
- **Sophisticated similarity scoring** with confidence metrics
- **Comprehensive preprocessing** for optimal text normalization
- **Customizable training pipeline** for domain-specific adaptation

## 🛠️ Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up directories:
   ```bash
   python setup_dirs.py
   ```

## 🚀 Running the Server

Start the API server:
```bash
python run.py
```

The server will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔌 API Endpoints

### Health Check
```
GET /api/health
```
Response:
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

### Predict Similarity
```
POST /api/predict
```
Request body:
```json
{
  "question1": "How do I learn Python?",
  "question2": "What's the best way to start with Python?"
}
```
Response:
```json
{
  "similarity_score": 0.87,
  "is_duplicate": true,
  "confidence": "High",
  "processing_time_ms": 42.5
}
```

## 🧠 Model Architecture

Our model uses a fine-tuned BERT architecture with:

- **Bidirectional context understanding** for capturing question semantics
- **Custom classification head** with multiple neural layers
- **Batch normalization** for training stability
- **Dropout regularization** to prevent overfitting
- **Gradient clipping** to handle exploding gradients
- **Learning rate scheduling** with warmup for optimal convergence

## 🔧 Training the Model

To train the model on your own data:

1. Ensure your dataset is in the `data` directory (should have columns: question1, question2, is_duplicate)
2. Run the training script:
   ```bash
   cd app/model
   python train.py
   ```

The script supports various parameters:
- Dataset size (default: 90,000 samples)
- Number of epochs (default: 8)
- Batch size (default: 16)
- Learning rate (default: 2e-5)

3. The trained model will be saved as `best_model.pt` in the model directory.

## 📊 Performance Metrics

Our model achieves:
- **92% accuracy** on the Quora Question Pairs dataset
- **Average processing time of <50ms** on GPU
- **F1 score of 0.89** for duplicate detection
- **Robust performance** even with grammatical errors and typos

## 🔍 Debugging

If you encounter issues:
1. Check the logs in the console
2. Verify GPU availability with `torch.cuda.is_available()`
3. Ensure all dependencies are correctly installed
4. Check that the model file exists at the expected location

---

Made with ❤️ by the Quora Query Matcher Team 