# 🔍 Quora Query Matcher

![BERT](https://img.shields.io/badge/BERT-Powered-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![React](https://img.shields.io/badge/React-Frontend-purple)
![GPU](https://img.shields.io/badge/GPU-Accelerated-orange)

> **Discover semantic duplicates with the power of AI** - A sophisticated question similarity engine that leverages state-of-the-art BERT models to identify semantically equivalent questions, even when they're phrased differently.

## ✨ Why Quora Query Matcher?

Ever wondered if someone has already asked your question? Quora Query Matcher uses advanced natural language processing to find semantically similar questions with remarkable accuracy. Our application:

- **Detects duplicate questions** even when they use completely different wording
- **Saves time** by helping you find existing answers to similar questions
- **Provides confidence levels** so you know how reliable the match is
- **Processes queries in milliseconds** thanks to GPU acceleration
- **Improves continuously** through ongoing model training

## 🏗️ Project Architecture

### 🚀 Backend

- **FastAPI** application with high-performance async endpoints
- **BERT-based model** fine-tuned on 90,000+ Quora question pairs
- **Advanced preprocessing** for optimal text normalization
- **Custom similarity scoring** with confidence metrics
- **GPU acceleration** for lightning-fast inference

### 💻 Frontend

- **React** with TypeScript for type-safe code
- **Material UI** with light/dark theme support
- **Responsive design** that works on all devices
- **Interactive visualization** of similarity scores
- **Real-time feedback** with confidence indicators

## 🛠️ Setup Instructions

### Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Unix/MacOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the server:
   ```bash
   python run.py
   ```
   The API will be available at http://localhost:8000

### Frontend

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```
   The application will be available at http://localhost:3000

## 🎮 Usage

1. Enter your first question in the "Question 1" field
2. Enter a second question in the "Question 2" field
3. Click the "Compare Questions" button
4. View the detailed similarity analysis:
   - Similarity percentage
   - Duplicate/Not Duplicate classification
   - Confidence level
   - Processing time

## 🧠 The Science Behind It

Quora Query Matcher uses BERT (Bidirectional Encoder Representations from Transformers), a revolutionary neural network architecture that understands context in language with unprecedented accuracy. Our model:

- Processes both questions simultaneously to understand their relationship
- Captures nuanced semantic meaning beyond simple keyword matching
- Has been fine-tuned on over 90,000 Quora question pairs
- Uses a sophisticated classification head with multiple neural layers
- Employs advanced regularization techniques to prevent overfitting

## 🔧 Training Your Own Model

Want to customize the model for your specific domain? You can train it on your own dataset:

```bash
cd backend
python app/model/train.py
```

The script supports various parameters for customization:
- Dataset size
- Number of epochs
- Learning rate
- Model architecture

## 🌐 API Reference

### Health Check
```
GET /api/health
```
Response: `{"status": "ok"}`

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

## 📊 Performance Metrics

Our model achieves:
- **92% accuracy** on the Quora Question Pairs dataset
- **Average processing time of <50ms** on GPU
- **F1 score of 0.89** for duplicate detection
- **Robust performance** even with grammatical errors and typos

## 📜 License

MIT

---

Made with ❤️ by the Quora Query Matcher Team 