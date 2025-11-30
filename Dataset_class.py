import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, dataframe, vocab_map, max_len=229):
        self.data = dataframe
        self.vocab = vocab_map
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['clean_text']
        label = self.data.iloc[idx]['label_idx']
        
        tokens = [self.vocab.get(word, self.vocab['<UNK>']) for word in text.split()]
        
        if len(tokens) < self.max_len:
            tokens = [0] * (self.max_len - len(tokens)) + tokens 
            
        else:
            tokens = tokens[-self.max_len:] 
            
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)