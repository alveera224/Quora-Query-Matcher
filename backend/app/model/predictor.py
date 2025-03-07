import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
import sys
import re

# Try importing the model, but allow fallback
try:
    from .train import SimilarityModel, clean_text
except ImportError:
    print("Could not import SimilarityModel, will use pretrained BERT instead")
    SimilarityModel = None
    
    # Define clean_text function if import fails
    def clean_text(text):
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

# Define old model architecture for compatibility
class OldSimilarityModel(torch.nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased"):
        super(OldSimilarityModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(768, 1)  # Original simple architecture
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x

class QuestionSimilarityPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.use_trained_model = False
        
        # Path to the saved model
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model.pt')
        
        # Try to load the trained model if it exists
        if os.path.exists(model_path):
            try:
                print(f"Loading trained model from {model_path}...")
                
                # Try with old architecture first since that's what we have saved
                try:
                    self.model = OldSimilarityModel(self.model_name)
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.to(self.device)
                    self.model.eval()
                    self.use_trained_model = True
                    print(f"Successfully loaded trained model with old architecture")
                except Exception as e:
                    print(f"Error loading with old architecture: {e}")
                    
                    # If old architecture fails and SimilarityModel is available, try new architecture
                    if SimilarityModel is not None:
                        try:
                            self.model = SimilarityModel(self.model_name)
                            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                            self.model.to(self.device)
                            self.model.eval()
                            self.use_trained_model = True
                            print(f"Successfully loaded trained model with new architecture")
                        except Exception as e2:
                            print(f"Error loading with new architecture: {e2}")
                            print("Using pre-trained BERT model instead")
                            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
                    else:
                        print("Using pre-trained BERT model instead")
                        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using pre-trained BERT model instead")
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        else:
            print(f"No trained model found at {model_path}")
            print("Using pre-trained BERT model instead")
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            
        self.max_length = 128
        print(f"Model initialized, using {'trained model' if self.use_trained_model else 'pre-trained BERT'}")

    def predict(self, question1: str, question2: str) -> float:
        """
        Predict the similarity between two questions
        """
        # Normalize the input text
        question1 = clean_text(question1)
        question2 = clean_text(question2)
        
        # Quick check for exact matches or very short questions
        if question1 == question2:
            return 1.0
        
        # Check for very short questions that might not have enough context
        if len(question1.split()) < 3 or len(question2.split()) < 3:
            # For very short questions, be more conservative with similarity
            similarity_factor = 0.8
        else:
            similarity_factor = 1.0
        
        if self.use_trained_model:
            # Using trained model - tokenize both questions together
            inputs = self.tokenizer(
                question1,
                question2,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            ).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(inputs["input_ids"], inputs["attention_mask"])
                
            # Get the similarity score
            raw_score = torch.sigmoid(outputs).item()
            
            # Apply a scaling function to make the distribution more discriminative
            # This helps separate similar from dissimilar questions
            similarity_score = self._scale_similarity(raw_score)
        else:
            # Using pre-trained BERT - get embeddings for each question separately
            embedding1 = self._get_embeddings(question1)
            embedding2 = self._get_embeddings(question2)
            
            # Calculate cosine similarity
            raw_score = self._cosine_similarity(embedding1, embedding2)
            
            # Apply scaling to the cosine similarity
            similarity_score = self._scale_similarity(raw_score)
        
        # Apply the similarity factor for short questions
        return similarity_score * similarity_factor
    
    def _normalize_text(self, text):
        """Normalize text by removing extra whitespace and converting to lowercase"""
        return ' '.join(text.lower().split())
    
    def _scale_similarity(self, raw_score):
        """
        Scale the similarity score to better distinguish between similar and dissimilar questions.
        This function applies a non-linear transformation to spread out the scores.
        """
        # Apply sigmoid-like scaling to push scores toward extremes
        if raw_score > 0.8:
            # High similarity scores get pushed higher
            return min(1.0, 0.8 + (raw_score - 0.8) * 2)
        elif raw_score < 0.3:
            # Low similarity scores get pushed lower
            return raw_score * 0.5
        else:
            # Middle range gets a slight adjustment
            return raw_score
    
    def _get_embeddings(self, text):
        """Get embeddings for the text using pre-trained BERT"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use CLS token embedding as sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings
    
    def _cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        similarity = np.dot(embedding1, embedding2.T) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return similarity[0][0] 