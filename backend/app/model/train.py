import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import os
import json
import random
import re
from tqdm import tqdm

class SimilarityModel(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", dropout_rate=0.2):
        super(SimilarityModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Freeze some of the BERT layers to prevent overfitting
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
            
        for i, layer in enumerate(self.bert.encoder.layer):
            # Only fine-tune the top 4 layers
            if i < 8:  # Freeze the first 8 layers
                for param in layer.parameters():
                    param.requires_grad = False
        
        # More sophisticated classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, input_ids, attention_mask):
        # Process through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Process through classification head
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

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

def augment_data(questions1, questions2, labels, augment_ratio=0.2):
    """Augment data with simple techniques like swapping questions"""
    aug_q1 = []
    aug_q2 = []
    aug_labels = []
    
    # Only augment positive examples (duplicates)
    positive_indices = [i for i, label in enumerate(labels) if label == 1]
    num_to_augment = int(len(positive_indices) * augment_ratio)
    
    if num_to_augment > 0:
        print(f"Augmenting {num_to_augment} positive examples")
        
        # Randomly select positive examples to augment
        indices_to_augment = random.sample(positive_indices, num_to_augment)
        
        for idx in indices_to_augment:
            # Swap question1 and question2 for duplicate pairs
            aug_q1.append(questions2[idx])
            aug_q2.append(questions1[idx])
            aug_labels.append(labels[idx])
    
    # Combine original and augmented data
    combined_q1 = np.concatenate([questions1, np.array(aug_q1)])
    combined_q2 = np.concatenate([questions2, np.array(aug_q2)])
    combined_labels = np.concatenate([labels, np.array(aug_labels)])
    
    return combined_q1, combined_q2, combined_labels

def load_data(data_path, limit=90000):
    """Load and preprocess data from CSV file, limiting to specified number of rows."""
    print(f"Loading data from {data_path}, limiting to {limit} samples")
    
    # Read CSV with proper handling of mixed data types
    df = pd.read_csv(data_path, low_memory=False)
    
    # Clean and validate the data
    print(f"Original data shape: {df.shape}")
    
    # Make sure required columns exist
    required_cols = ['question1', 'question2', 'is_duplicate']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for proper data types and handle errors
    try:
        # Try to convert is_duplicate to numeric
        df['is_duplicate'] = pd.to_numeric(df['is_duplicate'], errors='coerce')
        
        # Drop rows with invalid is_duplicate values
        invalid_labels = df['is_duplicate'].isna()
        if invalid_labels.any():
            print(f"Dropping {invalid_labels.sum()} rows with invalid labels")
            df = df[~invalid_labels]
    except Exception as e:
        print(f"Error processing labels: {e}")
        raise
    
    # Clean text data
    print("Cleaning text data...")
    df['question1'] = df['question1'].apply(clean_text)
    df['question2'] = df['question2'].apply(clean_text)
    
    # Remove rows with empty questions after cleaning
    empty_mask = (df['question1'].str.len() < 5) | (df['question2'].str.len() < 5)
    if empty_mask.any():
        print(f"Dropping {empty_mask.sum()} rows with empty or very short questions")
        df = df[~empty_mask]
    
    # Balance the dataset to have equal positive and negative examples
    positive_df = df[df['is_duplicate'] == 1]
    negative_df = df[df['is_duplicate'] == 0]
    
    # Determine the smaller class size
    min_class_size = min(len(positive_df), len(negative_df))
    
    # Limit each class to the smaller size or the specified limit/2
    max_per_class = min(min_class_size, limit // 2)
    
    # Sample from each class
    positive_df = positive_df.sample(max_per_class, random_state=42)
    negative_df = negative_df.sample(max_per_class, random_state=42)
    
    # Combine the balanced dataset
    balanced_df = pd.concat([positive_df, negative_df])
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced dataset shape: {balanced_df.shape}")
    
    # Extract data
    questions1 = balanced_df['question1'].astype(str).values
    questions2 = balanced_df['question2'].astype(str).values
    labels = balanced_df['is_duplicate'].values.astype(np.float32)
    
    # Augment data
    questions1, questions2, labels = augment_data(questions1, questions2, labels)
    
    print(f"Final data shape: Questions: {len(questions1)}, Labels: {len(labels)}")
    print(f"Label distribution: {np.bincount(labels.astype(int))}")
    
    return questions1, questions2, labels

def tokenize_data(tokenizer, questions1, questions2, max_length=128):
    """Tokenize question pairs."""
    inputs = []
    attention_masks = []
    
    for q1, q2 in tqdm(zip(questions1, questions2), total=len(questions1), desc="Tokenizing"):
        encoded = tokenizer.encode_plus(
            q1, q2,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        inputs.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    # Concatenate all input tensors
    inputs = torch.cat(inputs, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    return inputs, attention_masks

def train_model(model, train_dataloader, val_dataloader, device, epochs=8, learning_rate=2e-5):
    """Train the model with advanced techniques."""
    # Initialize optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    # Loss function with label smoothing for better generalization
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Calculate total training steps for scheduler
    total_steps = len(train_dataloader) * epochs
    
    # Create learning rate scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # 10% of total steps for warmup
        num_training_steps=total_steps
    )
    
    # Training stats
    best_val_loss = float('inf')
    training_stats = []
    
    # Save directory
    save_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print('-' * 40)
        
        # Training
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        for batch in progress_bar:
            # Unpack batch and move to device
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = loss_fn(outputs.squeeze(), labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs.squeeze(), labels)
                val_loss += loss.item()
                
                # Store predictions and true labels for metrics
                preds = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Calculate accuracy
        val_preds_binary = np.array(val_preds) >= 0.5
        val_true_binary = np.array(val_true) >= 0.5
        accuracy = np.mean(val_preds_binary == val_true_binary)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {accuracy:.4f}")
        
        # Save stats
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': float(accuracy)
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")
    
    # Save training stats
    with open(os.path.join(save_dir, 'training_stats.json'), 'w') as f:
        json.dump(training_stats, f)
    
    return model

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Set paths
    data_path = "../../data/q_quora.csv"
    
    # Load and preprocess data
    questions1, questions2, labels = load_data(data_path)
    
    # Split data
    q1_train, q1_val, q2_train, q2_val, y_train, y_val = train_test_split(
        questions1, questions2, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize data
    print("Tokenizing training data...")
    train_inputs, train_masks = tokenize_data(tokenizer, q1_train, q2_train)
    print("Tokenizing validation data...")
    val_inputs, val_masks = tokenize_data(tokenizer, q1_val, q2_val)
    
    # Convert labels to tensors
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)
    
    # Create DataLoaders
    print("Creating DataLoaders...")
    batch_size = 16
    
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    
    # Initialize model
    print("Initializing model...")
    model = SimilarityModel().to(device)
    
    # Train model
    print("Starting training...")
    model = train_model(model, train_dataloader, val_dataloader, device, epochs=8)
    
    print("Training complete!")

if __name__ == "__main__":
    main() 