import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import re
import json
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import nltk

from Dataset_class import EmotionDataset
from LSTM_class import EmotionBiLSTM

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE = 256
LEARNING_RATE = 0.005
EPOCHS = 30
MAX_LEN = 229
EMBED_DIM = 200

def load_and_clean_data():
    train_df = pd.read_csv('train.txt', names=['Text', 'Emotion'], sep=';')
    val_df = pd.read_csv('val.txt', names=['Text', 'Emotion'], sep=';')
    test_df = pd.read_csv('test.txt', names=['Text', 'Emotion'], sep=';')

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
        return " ".join(words)

    print("Cleaning text data...")
    train_df['clean_text'] = train_df['Text'].apply(clean_text)
    val_df['clean_text'] = val_df['Text'].apply(clean_text)
    test_df['clean_text'] = test_df['Text'].apply(clean_text)
    
    return train_df, val_df, test_df

train_df, val_df, test_df = load_and_clean_data()

all_text = " ".join(train_df['clean_text'].values)
words = all_text.split()
counter = Counter(words)
vocab = sorted(counter, key=counter.get, reverse=True)

vocab_to_int = {word: i+1 for i, word in enumerate(vocab)} 
vocab_to_int['<PAD>'] = 0
vocab_to_int['<UNK>'] = len(vocab_to_int)

print(f"Vocabulary Size: {len(vocab_to_int)}")

with open('vocab.json', 'w') as f:
    json.dump(vocab_to_int, f)
print("vocab.json saved successfully!")

vocab_size = len(vocab_to_int) + 1
embedding_matrix = np.zeros((vocab_size, EMBED_DIM))
glove_path = r'D:\Glove\glove.6B.200d.txt' 

print("Loading GloVe Embeddings...")
hits, misses = 0, 0
try:
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab_to_int:
                idx = vocab_to_int[word]
                vector = np.asarray(values[1:], dtype='float32')
                embedding_matrix[idx] = vector
                hits += 1
            else:
                misses += 1
    print(f"GloVe loaded: {hits} hits, {misses} misses")
except FileNotFoundError:
    print("WARNING: GloVe file not found! Using random embeddings instead.")

embedding_weights = torch.tensor(embedding_matrix, dtype=torch.float32).to(device)

le = LabelEncoder()
train_df['label_idx'] = le.fit_transform(train_df['Emotion'])
val_df['label_idx'] = le.transform(val_df['Emotion'])
test_df['label_idx'] = le.transform(test_df['Emotion'])

train_loader = DataLoader(EmotionDataset(train_df, vocab_to_int), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(EmotionDataset(val_df, vocab_to_int), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(EmotionDataset(test_df, vocab_to_int), batch_size=BATCH_SIZE, shuffle=False)

model = EmotionBiLSTM(vocab_size, EMBED_DIM, 6, weights=embedding_weights)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting Training...")
best_val_loss = float('inf')
patience = 4
counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

print("\nEvaluating on Test Set...")
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

target_names = le.inverse_transform(sorted(list(set(all_labels))))
print(classification_report(all_labels, all_preds, target_names=target_names))

torch.save(model.state_dict(), 'emotion_model.pth')
print("Model saved successfully!")