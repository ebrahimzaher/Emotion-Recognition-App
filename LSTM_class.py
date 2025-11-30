import torch
import torch.nn as nn

class EmotionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim, weights=None):
        super(EmotionBiLSTM, self).__init__()
        
        if weights is not None:
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.lstm1 = nn.LSTM(embed_dim, 256, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 128, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(128*2, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm1(embedded)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.dropout(out)
        out, (hidden, cell) = self.lstm3(out)
        
        cat_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        out = self.dropout(cat_hidden)
        out = self.fc(out)
        return out