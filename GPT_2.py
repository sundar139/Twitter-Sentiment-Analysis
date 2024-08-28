import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model, AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch import nn

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

sns.set(style='whitegrid')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 900)
pd.set_option('display.max_colwidth', 200)
warnings.filterwarnings("ignore")

def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = ['Tweet ID', 'Entity', 'Sentiment', 'Tweet Content']
    print(f"Loaded data from {file_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(df.head())
    print(f"Null values:\n{df.isnull().sum()}")
    return df

def preprocess_data(df):
    print("\nPreprocessing data:")
    print("Original dataframe shape:", df.shape)
    print("Unique values in Sentiment column:", df['Sentiment'].unique())
    
    sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2, 'Irrelevant': 3}
    df['label'] = df['Sentiment'].map(sentiment_mapping)
    
    print("Number of NaN labels:", df['label'].isna().sum())
    
    # Remove rows with NaN labels
    df = df.dropna(subset=['label'])
    
    print("Dataframe shape after removing NaN labels:", df.shape)
    
    if df.empty:
        raise ValueError("All data has been filtered out. Please check your input data.")
    
    # Convert labels to integers
    df['label'] = df['label'].astype(int)
    
    print("Final label distribution:", df['label'].value_counts())
    
    return df['Tweet Content'].tolist(), df['label'].tolist()

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(train_texts, train_labels, test_texts, test_labels, batch_size=16):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 does not have a pad token by default
    
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

    print(f"\nNumber of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    if len(train_dataset) > 0 and len(test_dataset) > 0:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print("DataLoaders created successfully.")
        return train_loader, test_loader
    else:
        print("Error: Empty dataset. Unable to create DataLoaders.")
        return None, None

class GPT2ForSequenceClassification(nn.Module):
    def __init__(self, num_labels=4):
        super(GPT2ForSequenceClassification, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, -1, :]  # Use the last token's hidden state
        
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return logits, loss

def train_model(model, train_loader, test_loader, num_epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()

        train_accuracy = correct_train / total_train
        train_loss = train_loss / len(train_loader)

        # Evaluation on test set
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        test_preds = []
        test_true = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits, loss = model(input_ids, attention_mask=attention_mask, labels=labels)

                test_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)

                test_preds.extend(preds.cpu().numpy())
                test_true.extend(labels.cpu().numpy())

        test_accuracy = correct_test / total_test
        test_loss = test_loss / len(test_loader)

        lr = scheduler.get_last_lr()[0]

        print(f'Epoch {epoch + 1}/{num_epochs} - LR: {lr:.6f} - '
              f'Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.4f} - '
              f'Val Loss: {test_loss:.4f} - Val Acc: {test_accuracy:.4f}')

    return test_true, test_preds

def main():
    # Load datasets
    train = load_data('Twitter Sentiment Analysis/twitter_training.csv')
    validation = load_data('Twitter Sentiment Analysis/twitter_validation.csv')

    # Preprocess data
    print("\nTraining data:")
    train_texts, train_labels = preprocess_data(train)
    print("\nValidation data:")
    test_texts, test_labels = preprocess_data(validation)

    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_texts, train_labels, test_texts, test_labels)

    if train_loader is None or test_loader is None:
        print("Error creating data loaders. Exiting.")
        return

    # Model setup
    model = GPT2ForSequenceClassification(num_labels=4)
    model.to(device)

    # Train the model
    test_true, test_preds = train_model(model, train_loader, test_loader)

    # Final evaluation
    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, target_names=['Negative', 'Neutral', 'Positive', 'Irrelevant']))

    # Calculate and print overall accuracy
    overall_accuracy = accuracy_score(test_true, test_preds)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'Twitter Sentiment Analysis/GPT2_Model.pth')
    print("Model saved.")

    # Confusion matrix
    labels = ['Negative', 'Neutral', 'Positive', 'Irrelevant']
    confusion_matrix_output = confusion_matrix(test_true, test_preds)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_output, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix - GPT-2 Model")
    plt.show()

if __name__ == '__main__':
    main()
