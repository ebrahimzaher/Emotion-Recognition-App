import streamlit as st
import torch
import torch.nn as nn
import json
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Emotion Analyzer", page_icon="🧠")

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class EmotionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim):
        super(EmotionBiLSTM, self).__init__()
        
        dummy_weights = torch.zeros(vocab_size, embed_dim)
        self.embedding = nn.Embedding.from_pretrained(dummy_weights, freeze=True)
        
        self.lstm1 = nn.LSTM(embed_dim, 256, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 128, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(128*2, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm1(embedded)
        out, _ = self.lstm2(out)
        out, (hidden, cell) = self.lstm3(out)
        cat_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(cat_hidden)
        return out


@st.cache_resource
def load_resources():
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = len(vocab) + 1 

    
    model = EmotionBiLSTM(vocab_size=len(vocab)+1, embed_dim=200, output_dim=6)

    model.load_state_dict(torch.load('emotion_model_93acc.pth', map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    
    return model, vocab, device

def preprocess_text(text, vocab, max_len=229):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    
    tokens = [vocab.get(w, vocab.get('<UNK>', 1)) for w in words]
    
    if len(tokens) < max_len:
        tokens = [0] * (max_len - len(tokens)) + tokens
    else:
        tokens = tokens[-max_len:]
        
    return torch.tensor([tokens], dtype=torch.long)

st.title("🧠 Human Emotion Recognition")
st.write("This AI Model uses **Bi-LSTM** & **GloVe Embeddings** to detect emotions from text.")

try:
    model, vocab, device = load_resources()
    st.success("Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

user_input = st.text_area("How are you feeling right now?", height=100)

if st.button("Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        tensor_input = preprocess_text(user_input, vocab).to(device)
        
        with torch.no_grad():
            outputs = model(tensor_input)
            probs = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()

        emotion_map = {0: 'Joy 😊', 1: 'Sadness 😢', 2: 'Anger 😡', 
                       3: 'Fear 😨', 4: 'Love ❤️', 5: 'Surprise 😲'}
        
        result = emotion_map[prediction]
        
        st.markdown(f"### Prediction: **{result}**")
        st.write(f"Confidence: **{confidence*100:.2f}%**")
        
        st.write("---")
        st.write("Class Probabilities:")
        probs_np = probs.cpu().numpy()[0]
        for idx, score in enumerate(probs_np):
            st.progress(float(score), text=f"{emotion_map[idx]}: {score*100:.1f}%")