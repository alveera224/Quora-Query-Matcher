from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.model.predictor import QuestionSimilarityPredictor
import time

router = APIRouter()
model = QuestionSimilarityPredictor()

class QuestionPair(BaseModel):
    question1: str
    question2: str

class PredictionResponse(BaseModel):
    similarity_score: float
    is_duplicate: bool
    confidence: str
    processing_time_ms: float

@router.post("/predict", response_model=PredictionResponse)
async def predict_similarity(questions: QuestionPair):
    try:
        # Validate input
        if not questions.question1 or not questions.question2:
            raise ValueError("Both questions must be provided and non-empty")
        
        # Measure processing time
        start_time = time.time()
            
        # Get prediction
        similarity_score = model.predict(
            questions.question1,
            questions.question2
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Determine if duplicate with adjusted threshold
        is_duplicate = similarity_score >= 0.70
        
        # Determine confidence level
        if similarity_score >= 0.85:
            confidence = "Very High"
        elif similarity_score >= 0.70:
            confidence = "High"
        elif similarity_score >= 0.50:
            confidence = "Medium"
        elif similarity_score >= 0.30:
            confidence = "Low"
        else:
            confidence = "Very Low"
        
        return PredictionResponse(
            similarity_score=float(similarity_score),
            is_duplicate=is_duplicate,
            confidence=confidence,
            processing_time_ms=processing_time
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        ) 